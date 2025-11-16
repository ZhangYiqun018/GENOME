from dataclasses import dataclass, asdict, field
from typing import List, Optional
import json
from datetime import datetime
from src.base.base_method import CombineMethod
from loguru import logger

@dataclass
class BaseConfig:
    model_name_or_path: str
    pools: List[str]
    llm_base_url: List[str]
    tasks: List[str]
    test_tasks: List[str]
    task_weights: List[float]
    combine_method: str
    
    plot_enabled: bool
    early_stop: bool
    early_stop_iter: int
    seed: int
    # optional; keep as keyword-only to avoid ordering issues in subclasses
    workspace_prefix: Optional[str] = field(default=None, kw_only=True)
    
    def __post_init__(self):
        self.optimizer_time = datetime.now().isoformat()
        self.combine_method = CombineMethod(self.combine_method)
        
        # normalize task weights
        self.task_weights = [weight / sum(self.task_weights) for weight in self.task_weights]
        # if test_tasks is not specified, use tasks as test_tasks
        if self.test_tasks is None:
            self.test_tasks = self.tasks
        
    def save(self, path):
        if isinstance(self.combine_method, CombineMethod):
            self.combine_method = self.combine_method.value
        
        with open(f"{path}/config.json", "w") as f:
            json.dump(
                asdict(self), f, indent=4, ensure_ascii=False
            )
        
    
    def validate(self):
        """Validate common configuration parameters."""
        # Validate tasks and weights
        if len(self.tasks) == 0:
            raise ValueError("Must specify at least one task.")
        if len(self.tasks) != len(self.task_weights):
            raise ValueError("Number of tasks must match number of task weights")
        if not all(0 <= w <= 1 for w in self.task_weights):
            raise ValueError("Task weights must be between 0 and 1")
        if abs(sum(self.task_weights) - 1) > 1e-6:
            raise ValueError("Task weights must sum to 1")
            
        # Validate LLM settings
        if not self.model_name_or_path:
            raise ValueError("Model name or path must be specified")
        if not self.llm_base_url:
            raise ValueError("LLM base URL must be specified")
        if not self.pools:
            raise ValueError("LoRA pools must be specified")
            
        # Validate early stopping
        if self.early_stop and self.early_stop_iter <= 0:
            raise ValueError("Early stop iteration must be positive when early stop is enabled")
            
        # Validate seed
        if self.seed < 0:
            raise ValueError("Seed must be non-negative")
