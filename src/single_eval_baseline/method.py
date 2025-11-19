from loguru import logger

from src.base.base_method import BaseMethod
from src.single_eval_baseline.config import SingleEvalBaselineConfig
from src.utils import load_lora_weight
from src.genome.individual import Individual


class SingleEvalBaselineMethod(BaseMethod):
    """Baseline that loads a single adapter and evaluates it on configured tasks."""

    def __init__(self, config: SingleEvalBaselineConfig):
        self.baseline_config = config
        super().__init__(config)
        config.validate()

        lora_path = self.pools[0]
        weights = load_lora_weight(lora_path)
        self.individual = Individual(
            id="single_eval",
            x=weights,
            parent=[lora_path],
            weight_path=lora_path,
            model_name_or_path=self.model_name_or_path,
            lora_config_path=lora_path,
        )

    def search(self):
        logger.info("Running single-eval baseline (no search/ensemble).")
        results = self.evaluate(individuals=[self.individual], split="valid")
        self.state["single_eval"] = results
        self.save_optim_state(state=self.state)
        return results
