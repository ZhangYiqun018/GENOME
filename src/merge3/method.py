import random
import time
from typing import List

from loguru import logger

from src.base.base_method import BaseMethod
from src.genome.individual import Individual
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

    def initialize(self) -> None:
        """Generate and evaluate an initial population before the search loop."""
        logger.info(
            f"Initializing Merge3 population with {self.merge3_config.nsga_population} candidates..."
        )
        start_time = time.time()
        self.individuals = []

        # 1) materialize nsga_population individuals
        for _ in range(self.merge3_config.nsga_population):
            genotype = [
                self.rng.uniform(*self.merge3_config.variable_bounds)
                for _ in range(self.merge3_config.genotype_dimension)
            ]
            parent_paths = self.problem.sample_parents()
            out_dir, merged_state = self.merger.materialize(genotype, parent_paths)
            individual = Individual(
                id=out_dir.split("_")[-1],
                x=merged_state,
                parent=list(parent_paths),
                weight_path=out_dir,
                lora_config_path=self.merger.lora_config_path,
                model_name_or_path=self.model_name_or_path,
            )
            individual.save_individual(out_dir)
            self.individuals.append(individual)

        # 2) evaluate all initialized individuals together
        weighted_scores = self.evaluate(individuals=self.individuals, split="valid")
        for individual_id, result in weighted_scores.items():
            self.update_global(
                id=individual_id,
                fitness_score=result["weighted_score"],
                path=result["path"],
                task_scores=result["task_scores"],
            )

        elapsed = time.time() - start_time
        self.update_optim_state(step=0, time=elapsed, weighted_scores=weighted_scores)
        self.save_optim_state(self.state)
        self.report_state(step=0)
        logger.info(
            f"Initialization completed: {len(self.individuals)} individuals, time {elapsed:.2f}s."
        )

    def search(self):
        logger.info(
            f"Starting Merge3 search: {self.merge3_config.nsga_generations} "
            f"generations x {self.merge3_config.nsga_population} population"
        )
        # Initial population evaluation (generation 0)
        self.initialize()

        best_individual = max(self.individuals, key=lambda ind: ind.fitness_score)
        best_score = best_individual.fitness_score

        # Main search loop starts from generation 1
        for generation in range(1, self.merge3_config.nsga_generations):
            start_time = time.time()
            logger.info(f"Generation {generation}")
            successful = 0
            for _ in range(self.merge3_config.nsga_population):
                genotype = [
                    self.rng.uniform(*self.merge3_config.variable_bounds)
                    for _ in range(self.merge3_config.genotype_dimension)
                ]
                try:
                    result = self.problem.evaluate_genotype(genotype)
                except Exception as exc:
                    logger.error(f"Failed to evaluate genotype {genotype}: {exc}")
                    continue
                self.individuals.append(result.individual)
                successful += 1
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
            if successful == 0:
                raise RuntimeError(
                    "Merge3 search aborted: no individuals evaluated successfully in "
                    f"generation {generation}. Check earlier errors before retrying."
                )
            self.update_optim_state(step=generation, time=elapsed)
            self.report_state(step=generation)

        if best_individual is None:
            logger.warning("Merge3 search produced no individual; aborting test phase")
            return

        self.ensemble_test(individuals=[best_individual], split="test")
