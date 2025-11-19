import os
from typing import List

from loguru import logger

from src.base.base_method import BaseMethod
from src.ensemble_baseline.config import EnsembleBaselineConfig
from src.genome.individual import Individual
from src.utils import load_lora_weight, save_lora_weight


class EnsembleBaselineMethod(BaseMethod):
    """Baseline that ensembles the provided pool adapters on the test set."""

    def __init__(self, config: EnsembleBaselineConfig):
        self.baseline_config = config
        super().__init__(config)
        config.validate()

        if self.baseline_config.pairwise_merge_before_ensemble:
            logger.info("Enabling pairwise merge of pool adapters before ensemble.")
            self.individuals: List[Individual] = self._build_pairwise_merged_pool()
        else:
            self.individuals: List[Individual] = self._load_pool_individuals()

    def _load_pool_individuals(self) -> List[Individual]:
        individuals: List[Individual] = []
        for path in self.pools:
            adapter_name = os.path.basename(path.rstrip("/")) or os.path.basename(path)
            weights = load_lora_weight(path)
            individual = Individual(
                id=adapter_name,
                x=weights,
                parent=[path],
                weight_path=path,
                model_name_or_path=self.model_name_or_path,
                lora_config_path=path,
            )
            individuals.append(individual)
            logger.info(f"Loaded pool adapter '{adapter_name}' from {path} for ensemble baseline.")
        return individuals

    def _build_pairwise_merged_pool(self) -> List[Individual]:
        """Merge randomly sampled pairs from the pool (deterministic by seed)."""
        num_pools = len(self.pools)
        n_pairs = self.baseline_config.pair_population
        logger.info(f"Pairwise merging {num_pools} pools into {n_pairs} merged adapters (seeded).")

        # reuse deterministic pair generation logic; samples/shuffles pairs with seed
        pairs = self.generate_pair_sequences(
            self.pools, n_samples=n_pairs, seed=self.baseline_config.seed
        )
        merged_individuals: List[Individual] = []

        out_root = os.path.join(self.workspace, "pairwise_merged")
        os.makedirs(out_root, exist_ok=True)

        for idx, (p1, p2) in enumerate(pairs):
            name1, name2 = os.path.basename(p1.rstrip("/")), os.path.basename(p2.rstrip("/"))
            merge_id = f"{name1}__{name2}"
            out_dir = os.path.join(out_root, f"pair_{idx:04d}_{merge_id}")

            weights = self.merge_lora_weights(
                lora_state_dicts=[load_lora_weight(p1), load_lora_weight(p2)],
                weights=[0.5, 0.5],
                method=self.combine_method,
            )
            save_lora_weight(
                lora_weight=weights,
                lora_path=out_dir,
                tokenizer=self.model_name_or_path,
                config=p1,
            )
            individual = Individual(
                id=merge_id,
                x=weights,
                parent=[p1, p2],
                weight_path=out_dir,
                model_name_or_path=self.model_name_or_path,
                lora_config_path=p1,
            )
            merged_individuals.append(individual)
            logger.info(f"Created merged adapter '{merge_id}' at {out_dir}")

        return merged_individuals

    def search(self):
        logger.info(
            "Starting ensemble baseline: running ensemble_test on all pool adapters."
        )
        return self.ensemble_test(individuals=self.individuals, split="test")
