from abc import ABC, abstractmethod
from enum import Enum
from openai import OpenAI
from vllm import LLM
import json
from typing import List, Dict

class Method(Enum):
    API = "api"
    LOCAL = "local"

class Split(Enum):
    TRAIN = "train"
    FULL = "full"
    VALID = "valid"
    TEST = "test"

class CombineMethod(Enum):
    TIES = "ties"
    LINEAR = "linear"
    DARE_TIES = "dare_ties"
    DARE_LINEAR = "dare_linear"
    BLXALPHA = "blxalpha"
    RANDOM = "random"

class Evaluator(ABC):
    def __init__(self):
        pass

    def load_jsonl(self, file_path: str) -> List[Dict]:
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
        return data
    
    @abstractmethod
    def load_data(self, split: str):
        pass
    
    def evaluate(self, method: str, max_samples: int | None = None, **kwargs):
        if method == Method.API:
            return self.api_evaluate(max_samples=max_samples, **kwargs)
        elif method == Method.LOCAL:
            return self.local_evaluate(max_samples=max_samples, **kwargs)
        else:
            raise ValueError(f"Invalid method: {method}")
    
    def load_model(self, model_name_or_path: str) -> LLM:
        llm = LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            trust_remote_code=True,
            enable_lora=True,
            device="auto",
            enforce_eager=True,
            tensor_parallel_size=1,
        )
        
        return llm
