from src.evaluate.eval import Evaluator, Split, Method
from openai import OpenAI
import os
import re
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from src.utils import get_gemma_prompt, get_llama3_1_prompt
from typing import List, Dict
from datasets import load_dataset, Dataset, disable_progress_bars
from loguru import logger
from src.deploy_vllm import online_load_lora, online_unload_lora
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from src.evaluate.MBPP.utils import imports, sanitize
from src.evaluate.MBPP.execution import check_correctness
import math

disable_progress_bars()

DATA_DIR = "datas/mbpp"

PROMPT = """You are an expert Python programmer, and here is your task:
{question}

Your code should pass these tests:
{test}
""".strip()

class MBPP(Evaluator):
    def __init__(self):
        self.task = "mbpp"
        self.seed = 42
        self.imports = imports
    
    def load_jsonl(self, path: str) -> List[Dict]:
        with open(path, "r") as f:
            return [json.loads(line) for line in f]
    
    def load_data(self, split: str, max_samples: int | None = None) -> Dataset:
        split = Split(split)
        if split == Split.TEST:
            data = self.load_jsonl(os.path.join(DATA_DIR, "test.json"))
        elif split == Split.VALID:
            data = self.load_jsonl(os.path.join(DATA_DIR, "valid.json"))
        else:
            raise ValueError(f"Invalid split: {split}")
        
        data = Dataset.from_list(data)
        if max_samples is not None and len(data) > max_samples:
            logger.warning(f"[MBPP] Capping valid set from {len(data)} to {max_samples}")
            data = data.select(range(max_samples))
        data = data.map(self.format_prompt)
        return data
    
    def format_prompt(self, item: Dict) -> Dict:
        prompt = PROMPT.format(
            question=item['text'],
            test="\n".join(item['test_list'])
        )
        return {"prompt": prompt}
        
    def extract_answer(self, text: str, test_list: List[str]) -> str:
        extract_code = sanitize(text)
        code = "\n".join(self.imports) + "\n" + extract_code + "\n" + "\n".join(test_list)
        
        return code
    
    def api_evaluate(self, llm: OpenAI, lora_name: str, lora_path: str, split: str, calculate_ppl: bool=False, return_predictions: bool=False, max_samples: int | None = None, **kwargs):
        def single_request(messages: List, lora_name: str, task_id: str, test_list: List[str], index: int):   
            response = llm.chat.completions.create(
                model=lora_name,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                seed=self.seed,
                max_tokens=4096,
                logprobs=calculate_ppl,
            )
            output = response.choices[0].message.content
            code = self.extract_answer(text=output, test_list=test_list)
            # correct
            result = check_correctness(
                task_id=task_id, completion_id=0,
                solution=code, time_out=10.0
            )['passed']
            results = dict(
                index=index,
                predicted_answer=result,
                reference_answer=True,
                is_correct=result,
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
                    logger.error(f"Error in single_request: {e}, index: {index}, messages: {messages}.")
                    raise e
                results["perplexity"] = ppl_value
            return results

        predictions = dict()    
        counter = 0
        ppls = 0
        data = self.load_data(split=split, max_samples=max_samples)
        if lora_path is not None:
            online_load_lora(
                base_url=llm.base_url,
                lora_name=lora_name,
                lora_path=lora_path,
            )
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = []
            for idx, item in enumerate(data):
                messages = [{"role": "user", "content": item['prompt']}]
                futures.append(
                    executor.submit(
                        single_request, 
                        messages=messages, 
                        lora_name=lora_name, 
                        task_id=item['task_id'], 
                        test_list=item['test_list'],
                        index=idx
                    )
                )
            for future in as_completed(futures):
                try:
                    result = future.result()  
                    if result['is_correct']:
                        counter += 1
                    
                    predictions[result['index']] = result
                    ppls += result.get("perplexity", 0)
                except Exception as e:
                    from loguru import logger
                    logger.error(f"Error: {e}")
                    predictions[result['index']] = None
        if lora_path is not None:
            online_unload_lora(
                base_url=llm.base_url,
                lora_name=lora_name,
            )
        results = {
            "score": counter / len(data),
        }
        if calculate_ppl:
            results["perplexity"] = ppls / len(predictions)
        if return_predictions:
            results["predictions"] = predictions
        return results

    def evaluate(self, method: str, max_samples: int | None = None, **kwargs):
        if method == Method.API:
            return self.api_evaluate(max_samples=max_samples, **kwargs)
        elif method == Method.LOCAL:
            return self.local_evaluate(max_samples=max_samples, **kwargs)
        else:
            raise ValueError(f"Invalid method: {method}")
        
        
    def local_evaluate(self, model_name_or_path: str, lora_path: str, split: str, max_samples: int | None = None, **kwargs):
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            seed=self.seed,
        )
        data = self.load_data(split=split, max_samples=max_samples)
        llm: LLM = self.load_model(model_name_or_path=model_name_or_path)
        
        batch_size = 1024
        counter = 0
        results = []
        for i in range(0, len(data), batch_size):
            batches = []
            test_lists = []
            for j in range(i, min(i + batch_size, len(data))):
                prompt = data[j]['prompt']
                test_lists.append(data[j]['test_list'])

                if "llama" in model_name_or_path.lower():
                    text = get_llama3_1_prompt(user_question=prompt)
                else:
                    text = get_gemma_prompt(user_question=prompt)
                batches.append(text)

            if lora_path == "base":
                outputs = llm.generate(batches, sampling_params=sampling_params)
            else:
                outputs = llm.generate(batches, sampling_params=sampling_params, lora_request=LoRARequest(lora_name=f"mbpp_{lora_path.split('/')[-1]}", lora_int_id=1, lora_path=lora_path))
            
            for idx, output in enumerate(outputs):
                code = self.extract_answer(text=output.outputs[0].text, test_list=test_lists[idx])
                result = check_correctness(
                    task_id=data[i + idx]['task_id'], completion_id=0,
                    solution=code, time_out=10.0
                )['passed']
                if result:
                    counter += 1
                results.append(dict(
                    prompt=batches[idx],
                    predict=output.outputs[0].text,
                    predicted_answer=code,
                    is_correct=result
                ))
        os.makedirs("results/mbpp", exist_ok=True)
        with open(f"results/mbpp/{lora_path.split('/')[-1]}_{split}_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        return counter / len(data)
