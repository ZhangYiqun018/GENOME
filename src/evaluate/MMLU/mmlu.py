from src.evaluate.eval import (
    Evaluator,
    Split,
    Method
)
from openai import OpenAI
import pandas as pd
import os
import re
from typing import (
    Dict,
    List,
    Any
)
from datasets import Dataset
from src.deploy_vllm import (
    online_load_lora,
    online_unload_lora
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import disable_progress_bars
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from src.utils import get_gemma_prompt
import json
from loguru import logger
import time
import math

disable_progress_bars()

DATA_DIR="datas/mmlu"

PROMPT = """Answer the following {subject} question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.

{question}

A. {A}
B. {B}
C. {C}
D. {D}

Let's think step by step."""

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"

class MMLU(Evaluator):
    def __init__(self):
        self.task = "mmlu"
        self.seed = 42

    def load_data(self, split: str) -> Dataset:
        split = Split(split)
        if split == Split.TEST:
            data = pd.read_csv(os.path.join(DATA_DIR, f"test.csv"))
        elif split == Split.FULL or split == Split.TRAIN:
            data = pd.read_csv(os.path.join(DATA_DIR, f"mmlu.csv"))
        elif split == Split.VALID:
            data = pd.read_csv(os.path.join(DATA_DIR, f"valid.csv"))
        else:
            raise ValueError(f"Invalid split: {split}")
        
        data = Dataset.from_pandas(data)
        data = data.map(lambda x: self.format_prompt(x))
        return data
    
    def format_prompt(self, item: Dict) -> Dict:
        prompt = PROMPT.format(
            subject = item["Subject"],
            question = item["Question"],
            A = item["A"],
            B = item["B"],
            C = item["C"],
            D = item["D"]
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
    
    def api_evaluate(self, llm: OpenAI, lora_name: str, lora_path: str, split: str, calculate_ppl: bool=False, return_predictions: bool=False, max_samples: int | None = None, **kwargs) -> Dict[str, Any]:
        def single_request(messages: List, lora_name: str, reference_answer: str, index: int):
            counter = 1
            while True:
                try:
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
                except Exception as e:
                    if counter > 0 and counter % 5 == 0:
                        logger.warning(f"This is the {counter}th time to get response, error message: {e}, retry at {min(counter, 10)} seconds ...")
                    time.sleep(min(counter, 10))
                    counter += 1
                    continue
        
        counter = 0
        data = self.load_data(split=split)
        if max_samples is not None and len(data) > max_samples:
            logger.warning(f"[MMLU] Capping valid set from {len(data)} to {max_samples}")
            data = data.select(range(max_samples))
        if lora_path is not None:   
            online_load_lora(
                base_url=llm.base_url,
                lora_name=lora_name,
                lora_path=lora_path,
            )
        predictions = dict()
        ppls = 0
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for idx, item in enumerate(data):
                messages = [{"role": "user", "content": item["prompt"]}]
                futures.append(
                    executor.submit(
                        single_request, 
                        messages=messages, 
                        lora_name=lora_name, 
                        reference_answer=item["Answer"],
                        index=idx
                    )
                )
            for future in as_completed(futures):
                result = future.result()
                if result['is_correct']:
                    counter += 1
                predictions[result['index']] = result
                ppls += result.get("perplexity", 0)
        if lora_path is not None:
            online_unload_lora(
                base_url=llm.base_url,
                lora_name=lora_name,
            )
        results = {
            'score': counter / len(data),
        }
        if calculate_ppl:
            results["perplexity"] = ppls / len(data)
        if return_predictions:
            results['predictions'] = predictions
            
        return results
    
    
    def local_evaluate(self, model_name_or_path: str, lora_path: str | None, split: str, max_samples: int | None = None, **kwargs):
        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.75,
            max_tokens=1024,
            seed=self.seed,
        )
        data = self.load_data(split=split)
        if max_samples is not None and len(data) > max_samples:
            logger.warning(f"[MMLU] Capping valid set from {len(data)} to {max_samples}")
            data = data.select(range(max_samples))
        llm: LLM = self.load_model(model_name_or_path=model_name_or_path)
        
        batch_size=1024
        counter = 0
        results = []
        for i in range(0, len(data), batch_size):
            batches = []
            reference_answers = []
            for j in range(i, min(i + batch_size, len(data))):
                prompt = data[j]["prompt"]
                reference_answer = data[j]["Answer"]
                
                reference_answers.append(reference_answer)
                text = get_gemma_prompt(user_question=prompt)
                batches.append(text)
            
            if lora_path == "base":
                outputs = llm.generate(batches, sampling_params=sampling_params)
            else:
                outputs = llm.generate(
                    batches, sampling_params=sampling_params, 
                    lora_request=LoRARequest(
                        lora_name=f"mmlu_{lora_path.split('/')[-1]}", 
                        lora_int_id=1,
                        lora_path=lora_path,
                    )
                )

            for idx, output in enumerate(outputs):
                predicted_answer = self.extract_answer(text=output.outputs[0].text)
                if predicted_answer == reference_answers[idx]:
                    counter += 1
                
                item = dict(
                    prompt=batches[idx],
                    predict=output.outputs[0].text,
                    reference_answer=reference_answers[idx],
                    predicted_answer=predicted_answer,
                    is_correct=predicted_answer == reference_answers[idx]
                )
                results.append(item)    
        os.makedirs(f"results/{self.task}", exist_ok=True)
        with open(f"results/{self.task}/{lora_path.split('/')[-1]}_{split}_results.json", "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        return counter / len(data)
