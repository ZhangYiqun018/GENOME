import os
import uuid
from typing import List, Sequence

from loguru import logger

from src.base.base_method import BaseMethod
from src.utils import load_lora_weight, save_lora_weight


class LoRAMerger:
    """Utility that reuses BaseMethod.merge_lora_weights to materialize adapters."""

    def __init__(self, method: BaseMethod, lora_config_path: str):
        self.method = method
        self.lora_config_path = lora_config_path

    def materialize(self, genotype: Sequence[float], parent_paths: Sequence[str]) -> tuple[str, dict]:
        if len(parent_paths) < 2:
            raise ValueError("Merge3 requires at least two parent adapters")

        weights = list(genotype[: len(parent_paths)])
        if not any(weights):
            weights = [1.0] * len(parent_paths)

        logger.debug("Merging parents %s with weights %s", parent_paths, weights)
        state_dicts = [load_lora_weight(path) for path in parent_paths]
        merged_state = self.method.merge_lora_weights(
            lora_state_dicts=state_dicts,
            weights=weights,
            method=self.method.combine_method,
        )

        individual_id = uuid.uuid4().hex
        out_dir = os.path.join(self.method.workspace, f"individual_{individual_id}")
        os.makedirs(out_dir, exist_ok=True)
        save_lora_weight(
            lora_weight=merged_state,
            lora_path=out_dir,
            tokenizer=self.method.model_name_or_path,
            config=self.lora_config_path,
        )
        return out_dir, merged_state
