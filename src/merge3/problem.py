import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from loguru import logger

from src.genome.individual import Individual
from src.merge3.merger import LoRAMerger


@dataclass
class EvaluationResult:
    individual: Individual
    weighted_score: float
    task_scores: Dict[str, float]


class Merge3Problem:
    """Bridges Merge3-style genotypes to GENOME's Individuals and evaluators."""

    def __init__(
        self,
        merger: LoRAMerger,
        pools: Sequence[str],
        parent_sample_size: int,
        rng: random.Random,
    ) -> None:
        self.merger = merger
        self.pools = list(pools)
        self.parent_sample_size = parent_sample_size
        self.rng = rng

    def sample_parents(self) -> List[str]:
        if len(self.pools) < self.parent_sample_size:
            raise ValueError("Not enough pools to sample parents for Merge3")
        return self.rng.sample(self.pools, self.parent_sample_size)

    def evaluate_genotype(self, genotype: Sequence[float]) -> EvaluationResult:
        parent_paths = self.sample_parents()
        out_dir, merged_state = self.merger.materialize(genotype, parent_paths)

        individual = Individual(
            id=out_dir.split("_")[-1],
            x=merged_state,
            parent=list(parent_paths),
            weight_path=out_dir,
            model_name_or_path=self.merger.method.model_name_or_path,
            lora_config_path=self.merger.lora_config_path,
        )
        individual.save_individual(out_dir)

        scores = self.merger.method.evaluate(individuals=[individual])
        weighted_score = scores[individual.id]["weighted_score"]
        logger.info(
            "Merge3 genotype produced score %.4f on parents %s",
            weighted_score,
            parent_paths,
        )
        return EvaluationResult(
            individual=individual,
            weighted_score=weighted_score,
            task_scores=scores[individual.id]["task_scores"],
        )
