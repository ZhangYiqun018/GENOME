import random
import time
from typing import List

from loguru import logger

from src.base.base_method import BaseMethod
from src.merge3.config import Merge3Config
from src.merge3.merger import LoRAMerger
from src.merge3.problem import Merge3Problem


class Merge3LoRAMethod(BaseMethod):
    """Implements a Merge3-style search loop while reusing GENOME interfaces."""

    def __init__(self, config: Merge3Config):
        self.merge3_config = config
        super().__init__(config)
        config.validate()

        self.rng = random.Random(config.seed)
        self.merger = LoRAMerger(self, lora_config_path=self.pools[0])
        self.problem = Merge3Problem(
            merger=self.merger,
            pools=self.pools,
            parent_sample_size=config.parent_sample_size,
            rng=self.rng,
        )
        self.individuals: List = []

    def search(self):
        logger.info("Starting Merge3 search: %d generations x %d population",
                    self.merge3_config.nsga_generations,
                    self.merge3_config.nsga_population)
        best_individual = None
        best_score = float("-inf")

        for generation in range(self.merge3_config.nsga_generations):
            start_time = time.time()
            logger.info("Generation %d", generation + 1)
            for _ in range(self.merge3_config.nsga_population):
                genotype = [
                    self.rng.uniform(*self.merge3_config.variable_bounds)
                    for _ in range(self.merge3_config.genotype_dimension)
                ]
                result = self.problem.evaluate_genotype(genotype)
                self.individuals.append(result.individual)
                if result.weighted_score > best_score:
                    best_score = result.weighted_score
                    best_individual = result.individual
                    self.update_global(
                        id=result.individual.id,
                        fitness_score=result.weighted_score,
                        path=result.individual.weight_path,
                        task_scores=result.task_scores,
                    )
            elapsed = time.time() - start_time
            self.update_optim_state(step=generation, time=elapsed)
            self.report_state(step=generation)

        if best_individual is None:
            logger.warning("Merge3 search produced no individual; aborting test phase")
            return

        self.ensemble_test(individuals=[best_individual], split="test")
