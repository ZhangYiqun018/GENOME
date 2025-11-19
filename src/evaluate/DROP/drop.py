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
from src.evaluate.DROP.scoring import drop_metric
import json
import math

disable_progress_bars()

DATA_DIR = "datas/drop"

PROMPT= """You will be asked to read a passage and answer a question.
Write a line of the form "Answer: $ANSWER" at the end of your response.

{context}
""".strip()

# context, ref_text, completion

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

class DROP(Evaluator):
    def __init__(self):
        self.task = "drop"
        self.seed = 42

    
    def load_data(self, split: str, max_samples: int | None = None):
        split = Split(split)
        if split == Split.TEST:
            data = self.load_jsonl(os.path.join(DATA_DIR, "drop_v0_test.jsonl"))
        elif split == Split.VALID:
            data = self.load_jsonl(os.path.join(DATA_DIR, "drop_v0_validation.jsonl"))
        elif split == Split.FULL or split == Split.TRAIN:
            data = self.load_jsonl(os.path.join(DATA_DIR, "drop_v0_train.jsonl"))
        else:
            raise ValueError(f"Invalid split: {split}")
        
        data = Dataset.from_list(data)
        if max_samples is not None and len(data) > max_samples:
            logger.warning(f"[DROP] Capping valid set from {len(data)} to {max_samples}")
            data = data.select(range(max_samples))
        data = data.map(self.format_prompt)
        return data
    
    def format_prompt(self, item: Dict) -> Dict:
        context = item['context']        
        prompt = PROMPT.format(
            context=context
        )
        return {"prompt": prompt}
    
    def extract_answer(self, text: str) -> str:
        match = re.search(ANSWER_PATTERN, text)
        if match:
            return match.group(1).strip()
        return text
    def api_evaluate(self, llm: OpenAI, lora_name: str, lora_path: str, split: str, calculate_ppl: bool=False, return_predictions: bool=False, max_samples: int | None = None, **kwargs):
        def single_request(messages: List, lora_name: str, reference_answer: str, index: int):
            response = llm.chat.completions.create(
                model=lora_name,
                messages=messages,
                temperature=0.2,
                top_p=0.75,
                seed=self.seed,
                max_tokens=1024,
                logprobs=calculate_ppl,
            )
            output = response.choices[0].message.content
            predicted_answer = self.extract_answer(text=output)
            em_score, f1_score = drop_metric(predicted_answer, reference_answer)

            result = dict(
                index=index,
                predicted_answer=predicted_answer,
                reference_answer=reference_answer,
                is_correct=None,
                score = dict(
                    em_score=em_score,
                    f1_score=f1_score,
                ),
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
        online_load_lora(
            base_url=llm.base_url,
            lora_name=lora_name,
            lora_path=lora_path,
        )
        ppls = 0
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = []
            for idx, item in enumerate(data):
                messages = [{"role": "user", "content": item["prompt"]}]
                futures.append(
                    executor.submit(
                        single_request, 
                        messages=messages, 
                        lora_name=lora_name, 
                        reference_answer=item['ref_text'].split("|"),
                        index=idx
                    )
                )
            for future in as_completed(futures):
                result = future.result()
                if result:
                    predictions[result['index']] = result
                ppls += result.get("perplexity", 0)
                
        online_unload_lora(
            base_url=llm.base_url,
            lora_name=lora_name,
        )
        ave_em_score = sum([p['score']['em_score'] for p in predictions.values()]) / len(predictions)
        ave_f1_score = sum([p['score']['f1_score'] for p in predictions.values()]) / len(predictions)
        results = {
            'score': ave_em_score,
        }
        if calculate_ppl:
            results["perplexity"] = ppls / len(predictions)
        if return_predictions:
            results["predictions"] = predictions
        return results
    
    def local_evaluate(self, model_name_or_path: str, lora_path: str, split: str, max_samples: int | None = None, **kwargs):
        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.75,
            max_tokens=1024,
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
                reference_answer = data[j]["ref_text"].split("|")
                
                reference_answers.append(reference_answer)
                text = get_gemma_prompt(user_question=prompt)
                batches.append(text)
            
            if lora_path == "base":
                outputs = llm.generate(batches, sampling_params=sampling_params)
            else:
                outputs = llm.generate(
                    batches, sampling_params=sampling_params,
                    lora_request=LoRARequest(
                        lora_name=f"drop_{lora_path.split('/')[-1]}", 
                        lora_int_id=1,
                        lora_path=lora_path,
                    )
                )
                
            for idx, output in enumerate(outputs):
                predicted_answer = self.extract_answer(text=output.outputs[0].text)
                reference_answer = reference_answers[idx]

                em_score, f1_score = drop_metric(predicted_answer, reference_answer)
                
                results.append(dict(
                    prompt=batches[idx],
                    predict=output.outputs[0].text,
                    reference_answer=reference_answers[idx],
                    predicted_answer=predicted_answer,
                    is_correct=None,
                    score=dict(
                        em_score=em_score,
                        f1_score=f1_score
                    )
                ))
        
        os.makedirs("results/drop", exist_ok=True)
        with open(f"results/drop/{lora_path.split('/')[-1]}_{split}_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        

        em_scores = [result['score']['em_score'] for result in results]
        f1_scores = [result['score']['f1_score'] for result in results]

        return dict(
            em_score = sum(em_scores) / len(em_scores),
            f1_score = sum(f1_scores) / len(f1_scores)
        )
