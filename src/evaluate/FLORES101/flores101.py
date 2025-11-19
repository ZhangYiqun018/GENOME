from src.evaluate.eval import Evaluator, Split, Method
from openai import OpenAI
import os
import re
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from src.utils import get_gemma_prompt
from typing import List, Dict
from datasets import load_dataset, Dataset, disable_progress_bars
from loguru import logger
from src.deploy_vllm import online_load_lora, online_unload_lora
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import math

disable_progress_bars()


DATA_DIR = "datas/flores101"

ZEROSHOT_PROMPT = """Translate the following sentence from English to {language}, the last line of your response should be of the following format: 'Translation: $SENTENCE' (without quotes) where $SENTENCE is the translated sentence in {language}.

{sentence}
""".strip()

FEWSHOT_PROMPT = """Translate the following sentence from English to {language}. Your output should be formatted as follows:

Translation: $SENTENCE

(where $SENTENCE is the translated version of the sentence into {language}).

Below are examples to guide the translation task:

{examples}

Now translate the following sentence:

{sentence}
""".strip()

ANSWER_PATTERN = r"(?i)Translation\s*:\s*([^\n]+)"

class FLORES101(Evaluator):
    def __init__(self, model_name_or_path: str | None = None, language_numbers: int=37):
        self.language_numbers = language_numbers
        self.task = f"flores{self.language_numbers}"
        self.seed = 42
        # Temporary fallback: allow env override to avoid HF fetch failures.
        model_name_or_path = (
            os.getenv("GENOME_MODEL_PATH")
            or "/mnt/shared-storage-user/zhangyiqun/models/gemma-2-2b-it"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.number_shots: int = 3
        
    def load_data(self, split: str, max_samples: int | None = None) -> Dataset:
        split = Split(split)
        if split == Split.TEST:
            data = self.load_jsonl(file_path=os.path.join(DATA_DIR, f"flores{self.language_numbers}_template_test.json"))
        elif split == Split.VALID:
            data = self.load_jsonl(file_path=os.path.join(DATA_DIR, f"flores{self.language_numbers}_template_valid.json"))
        elif split == Split.FULL or split == Split.TRAIN:
            raise ValueError(f"Invalid split in {self.task}: {split}")
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # preprocess
        data = Dataset.from_list(data)
        if max_samples is not None and len(data) > max_samples:
            logger.warning(f"[FLORES{self.language_numbers}] Capping valid set from {len(data)} to {max_samples}")
            data = data.select(range(max_samples))
        data = data.map(self.format_prompt)
        return data
    
    def format_prompt(self, item: Dict) -> Dict:
        template_reference_list = item['template_reference_list']
        template_candidate_list = item['template_candidate_list']
        template = ""
        for idx, (reference, candidate) in enumerate(zip(template_reference_list, template_candidate_list)):
            template += f"{reference}\nTranslation: {candidate}\n\n"
            if idx == self.number_shots - 1:
                break
        
        prompt = FEWSHOT_PROMPT.format(sentence = item["reference"], language = item["language"], examples=template)
        
        return {"prompt": prompt}
     
    def extract_answer(self, text: str) -> str:
        match = re.search(ANSWER_PATTERN, text)
        if match:
            return match.group(1).strip()
        return ""
    
    def calcuate_bleu_score(self, predicted: str, reference: str) -> float:
        smooth = SmoothingFunction()
        predicted = self.tokenizer.tokenize(predicted)
        reference = self.tokenizer.tokenize(reference)
        bleu_score = sentence_bleu(references = [reference], hypothesis = predicted, smoothing_function=smooth.method1)
        
        return bleu_score
    
    def api_evaluate(self, llm: OpenAI, lora_name: str, lora_path: str, split: str, calculate_ppl: bool=False, return_predictions: bool=False, max_samples: int | None = None, **kwargs):
        def single_request(messages: List, lora_name: str, reference_answer: str, index: int):
            response = llm.chat.completions.create(
                model=lora_name,
                messages=messages,
                temperature=0.2,
                top_p=0.75,
                seed=self.seed,
                max_tokens=512,
                logprobs=calculate_ppl,
            )
            output = response.choices[0].message.content    
            predicted_answer = self.extract_answer(text=output)
            bleu_score = self.calcuate_bleu_score(predicted_answer, reference_answer)
            result = dict(
                index=index,
                predicted_answer=predicted_answer,
                reference_answer=reference_answer,
                score=dict(
                    bleu=bleu_score,
                ),
                is_correct=None,
                perplexity=0,
            )
            if calculate_ppl:
                try:
                    logprobs = response.choices[0].logprobs
                    logits = sum([log.logprob for log in logprobs.content])
                    ppl_value = math.exp(-logits / len(logprobs.content))
                    assert isinstance(ppl_value, float), f"ppl_value is not a float: {ppl_value}"
                except Exception as e:
                    from loguru import logger
                    logger.error(f"Error in calculating ppl: {e}")
                result["perplexity"] = ppl_value
            return result

        predictions = dict()
        data = self.load_data(split=split, max_samples=max_samples)
        ppls = 0
        
        online_load_lora(
            base_url=llm.base_url,
            lora_name=lora_name,
            lora_path=lora_path,
        )
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = []
            for idx, item in enumerate(data):
                messages = [{"role": "user", "content": item["prompt"]}]
                futures.append(
                    executor.submit(
                        single_request, 
                        messages=messages, 
                        lora_name=lora_name, 
                        reference_answer=item["candidate"],
                        index=idx
                    )
                )
            for future in as_completed(futures):
                result = future.result()
                predictions[result['index']] = result
                ppls += result.get("perplexity", 0)
        online_unload_lora(
            base_url=llm.base_url,
            lora_name=lora_name,
        )
        results = {
            "score": sum([item['score']['bleu'] for item in predictions.values()]) / len(predictions),
        }
        if calculate_ppl:
            results["perplexity"] = ppls / len(data)
        if return_predictions:
            results["predictions"] = predictions
            
        return results
        
    def local_evaluate(self, model_name_or_path: str, lora_path: str, split: str, max_samples: int | None = None, **kwargs) -> float:
        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.75,
            max_tokens=512,
            seed=self.seed,
        )
        data = self.load_data(split=split, max_samples=max_samples)
        llm: LLM = self.load_model(model_name_or_path=model_name_or_path)
        
        batch_size = 1024
        counter = 0
        results = []
        for i in range(0, len(data), batch_size):
            batches = []
            reference_answers = []
            
            for j in range(i, min(i + batch_size, len(data))):
                prompt = data[j]["prompt"]
                reference_answer = data[j]["candidate"]
                
                reference_answers.append(reference_answer)
                text = get_gemma_prompt(user_question=prompt)
                batches.append(text)

            if lora_path == "base":
                outputs = llm.generate(batches, sampling_params=sampling_params)
            else:
                outputs = llm.generate(
                    batches, sampling_params=sampling_params,
                    lora_request=LoRARequest(
                        lora_name=f"{self.task}_{lora_path.split('/')[-1]}", 
                        lora_int_id=1, 
                        lora_path=lora_path,
                    )
                )
                
            for idx, output in enumerate(outputs):
                predicted_answer = self.extract_answer(text=output.outputs[0].text)
                bleu_score = self.calcuate_bleu_score(predicted_answer, reference_answers[idx])
                item = dict(
                    prompt=batches[idx],
                    predict=output.outputs[0].text,
                    reference_answer=reference_answers[idx],
                    predicted_answer=predicted_answer,
                    score=dict(
                        bleu=bleu_score,
                    ),
                    is_correct=None,
                )
                results.append(item)
        
        ave_bleu_score = sum([item['score']['bleu'] for item in results]) / len(results)
        os.makedirs(f"results/{self.task}", exist_ok=True)
        with open(f"results/{self.task}/{lora_path.split('/')[-1]}_{split}_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        return ave_bleu_score
