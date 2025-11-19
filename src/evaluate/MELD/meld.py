from src.evaluate.eval import Evaluator, Split, Method
from openai import OpenAI
import pandas as pd
import os
import re
from typing import Dict, List, Any
from datasets import Dataset, disable_progress_bars
from loguru import logger
from src.deploy_vllm import online_load_lora, online_unload_lora
from vllm.lora.request import LoRARequest
from concurrent.futures import ThreadPoolExecutor, as_completed
from vllm import LLM, SamplingParams
import json
from src.utils import get_gemma_prompt
from sklearn.metrics import f1_score
import math 

disable_progress_bars()

DATA_DIR = "datas/meld"

PROMPT = """Given a conversation history and a current utterance, follow these steps to identify the emotion of the current utterance from the given options. The emotion should be determined based on both the conversation context and the current utterance.
The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDEFG. Let's think step by step.

History:
{history}

Utterance:
{utterance}

Options:
{options}"""

# ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"
ANSWER_PATTERN = r"(?i)Answer\s*:\s*([A-G])[.\s\n]?"

class MELD(Evaluator):
    def __init__(self):
        self.task = "meld"
        self.seed = 42
        
    def load_data(self, split: str, max_samples: int | None = None) -> Dataset:
        split = Split(split)
        if split == Split.TEST:
            data = self.load_jsonl(os.path.join(DATA_DIR, f"test.json"))
        elif split == Split.FULL or split == Split.TRAIN:
            data = self.load_jsonl(os.path.join(DATA_DIR, f"full.json"))
        elif split == Split.VALID:
            data = self.load_jsonl(os.path.join(DATA_DIR, f"valid.json"))
        else:
            raise ValueError(f"Invalid split: {split}")
        
        data = Dataset.from_list(data)
        if max_samples is not None and len(data) > max_samples:
            logger.warning(f"[MELD] Capping valid set from {len(data)} to {max_samples}")
            data = data.select(range(max_samples))
        data = data.map(lambda x: self.format_prompt(x))
        return data
    
    def format_prompt(self, item: Dict) -> Dict:
        prompt = PROMPT.format(
            history = "- "+"\n- ".join(item["history"]),
            utterance = item["utterance"],
            options = "\n".join([f"{key}. {value}" for key, value in item["candidate"].items()])
        )
        return {"prompt": prompt}
    
    def extract_answer(self, text: str) -> str:
        match = re.search(ANSWER_PATTERN, text)
        if match:
            return match.group(1).strip()
        return "C"
    
    def evaluate(self, method: str, max_samples: int | None = None, **kwargs):
        if method == Method.API:
            return self.api_evaluate(max_samples=max_samples, **kwargs)
        elif method == Method.LOCAL:
            return self.local_evaluate(max_samples=max_samples, **kwargs)
        else:
            raise ValueError(f"Invalid method: {method}")

    def api_evaluate(self, llm: OpenAI, lora_name: str, lora_path: str, split: str, calculate_ppl: bool=False, return_predictions: bool = False, max_samples: int | None = None, **kwargs):
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
            result = dict(
                index=index,
                predicted_answer=predicted_answer,
                reference_answer=reference_answer,
                is_correct=predicted_answer == reference_answer,
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
            
        data = self.load_data(split=split, max_samples=max_samples)
        ppls = 0
        online_load_lora(
            base_url=llm.base_url,
            lora_name=lora_name,
            lora_path=lora_path,
        )
        predictions = dict()
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = []
            for idx, item in enumerate(data):
                messages = [{"role": "user", "content": item["prompt"]}]
                futures.append(
                    executor.submit(
                        single_request, 
                        messages=messages, 
                        lora_name=lora_name, 
                        reference_answer=item["answer"], 
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
        
        sorted_predictions = dict(sorted(predictions.items(), key=lambda x: x[0]))
        y_true = [item['reference_answer'] for _, item in sorted_predictions.items()]
        y_pred = [item['predicted_answer'] for _, item in sorted_predictions.items()]
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')

        results = {
            'score': weighted_f1,
        }
        if calculate_ppl:
            results["perplexity"] = ppls / len(data)
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
        results = []
        for i in range(0, len(data), batch_size):
            batches = []
            reference_answers = []
            for j in range(i, min(i + batch_size, len(data))):
                prompt = data[j]["prompt"]
                reference_answer = data[j]["answer"]
                
                reference_answers.append(reference_answer)
                text = get_gemma_prompt(user_question=prompt)
                batches.append(text)
                
            if lora_path == "base":
                outputs = llm.generate(batches, sampling_params=sampling_params)
            else:
                outputs = llm.generate(
                    batches, sampling_params=sampling_params, 
                    lora_request=LoRARequest(
                        lora_name='meld_'+lora_path.split('/')[-1], 
                        lora_int_id=1,
                        lora_path=lora_path,
                    )
                )
            
            for idx, output in enumerate(outputs):
                predicted_answer = self.extract_answer(text=output.outputs[0].text)
                results.append(dict(
                    index=idx,
                    prompt=batches[idx],
                    predict=output.outputs[0].text,
                    predicted_answer=predicted_answer,
                    reference_answer=reference_answers[idx],
                ))
        os.makedirs("results/meld", exist_ok=True)
        with open(f"results/meld/{lora_path.split('/')[-1]}_{split}_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        y_true = [item['reference_answer'] for item in results]
        y_pred = [item['predicted_answer'] for item in results]
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        
        return weighted_f1
        
if __name__ == "__main__":
    evaluator = MELD()
    data = evaluator.load_data(Split.FULL)
    print(data)
    print(data[0]['prompt'])
    
