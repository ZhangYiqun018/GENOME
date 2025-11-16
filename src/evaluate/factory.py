from src.evaluate.MMLU import MMLU
from src.evaluate.GSM8k import GSM8k
from src.evaluate.ARC_C import ARC_C
from src.evaluate.MBPP import MBPP
from src.evaluate.CSQA import CSQA
from src.evaluate.MATH import Math
from src.evaluate.MELD import MELD
from src.evaluate.EMORYNLP import EMORYNLP
from src.evaluate.MMLUPro import MMLUPro
from src.evaluate.DROP import DROP
from src.evaluate.MGSM import MGSM
from src.evaluate.FLORES101 import FLORES101
from src.evaluate.BBH import BBH
from enum import Enum

class Benchmark(Enum):
    MMLU="mmlu"
    MMLU_PRO="mmlupro"
    MMLU_PRO_REASONING="mmluproreasoning"
    MMLU_PRO_KNOWLEDGE="mmluproknowledge"
    GSM8K="gsm8k"
    ARC_C="arcc"
    MBPP="mbpp"
    CSQA="csqa"
    MATH="math"
    MELD="meld"
    EMORYNLP="emorynlp"
    DROP="drop"
    MGSM="mgsm"
    FLORES101="flores101"
    FLORES37="flores37"
    TRUTHFULQA="truthfulqa"
    BBH="bbh"
    
class EvaluatorFactory:
    def __init__(self, model_name_or_path: str=None):
        if model_name_or_path is None:
            self.model_name_or_path = "gemma-2-2b-it"
        else:
            self.model_name_or_path = model_name_or_path
    
    def get_evaluator(self, task: str):
        if isinstance(task, str):
            task = Benchmark(task.lower())
            
        if not isinstance(task, Benchmark):
            raise TypeError(f"Task must be a string or Benchmark enum, got {type(task)}")
            
        if task == Benchmark.MMLU:
            return MMLU()
        elif task == Benchmark.MMLU_PRO:
            return MMLUPro(type="default")
        elif task == Benchmark.MMLU_PRO_REASONING:
            return MMLUPro(type="reasoning")
        elif task == Benchmark.MMLU_PRO_KNOWLEDGE:
            return MMLUPro(type="knowledge")
        elif task == Benchmark.GSM8K:
            return GSM8k()
        elif task == Benchmark.ARC_C:
            return ARC_C()
        elif task == Benchmark.MBPP:
            return MBPP()
        elif task == Benchmark.CSQA:
            return CSQA()
        elif task == Benchmark.MATH:
            return Math()
        elif task == Benchmark.MELD:
            return MELD()
        elif task == Benchmark.EMORYNLP:
            return EMORYNLP()
        elif task == Benchmark.DROP:
            return DROP()
        elif task == Benchmark.MGSM:
            return MGSM()
        elif task == Benchmark.FLORES101:
            return FLORES101(model_name_or_path=self.model_name_or_path, language_numbers=101)
        elif task == Benchmark.FLORES37:
            return FLORES101(model_name_or_path=self.model_name_or_path, language_numbers=37)
        elif task == Benchmark.TRUTHFULQA:
            try:
                from src.evaluate.TRUTHFULQA import TRUTHFULQA
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "TRUTHFULQA evaluator not available; missing `src.evaluate.TRUTHFULQA`. "
                    "Add the evaluator or remove truthfulqa from tasks."
                ) from exc
            return TRUTHFULQA()
        elif task == Benchmark.BBH:
            return BBH()
        else:
            raise ValueError(f"Evaluator for task {task} not found.")
