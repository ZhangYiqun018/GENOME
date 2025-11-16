import time
from typing import List, Optional

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from loguru import logger

from src.base.base_method import BaseMethod
from src.genome.individual import Individual
from src.merge3.config import Merge3Config
from src.merge3.merger import LoRAMerger
from src.merge3.pymoo_problem import Merge3PymooProblem


class Merge3LoRAMethod(BaseMethod):
    """Implements a Merge3-style search loop while reusing GENOME interfaces."""

    def __init__(self, config: Merge3Config):
        self.merge3_config = config
        super().__init__(config)
        config.validate()

        self.merger = LoRAMerger(self, lora_config_path=self.pools[0])
        self.individuals: List = []

    def search(self):
        logger.info(
            f"Starting Merge3 search (pymoo NSGA-II): {self.merge3_config.nsga_generations} "
            f"generations x {self.merge3_config.nsga_population} population"
        )

        if len(self.pools) < self.merge3_config.parent_sample_size:
            raise ValueError("Not enough pools to sample parents for Merge3")

        parent_paths = self.pools[: self.merge3_config.parent_sample_size]
        problem = Merge3PymooProblem(
            method=self,
            merger=self.merger,
            parent_paths=parent_paths,
            genotype_dimension=self.merge3_config.genotype_dimension,
            variable_bounds=self.merge3_config.variable_bounds,
        )
        algorithm = NSGA2(
            pop_size=self.merge3_config.nsga_population,
            eliminate_duplicates=True,
        )

        start_time = time.time()
        minimize(
            problem,
            algorithm,
            ("n_gen", self.merge3_config.nsga_generations),
            seed=self.merge3_config.seed,
            verbose=False,
        )
        elapsed = time.time() - start_time

        best_individual: Optional[Individual] = problem.best_individual
        if best_individual is None:
            logger.warning("Merge3 search produced no individual; aborting test phase")
            return

        best_score = problem.best_score
        best_task_scores = problem.best_task_scores or {}
        best_path = problem.best_path or best_individual.weight_path

        self.individuals = [best_individual]
        self.update_global(
            id=best_individual.id,
            fitness_score=best_score,
            path=best_path,
            task_scores=best_task_scores,
        )
        self.update_optim_state(
            step=self.merge3_config.nsga_generations - 1, time=elapsed
        )
        self.report_state(step=self.merge3_config.nsga_generations - 1)

        self.ensemble_test(individuals=[best_individual], split="test")
