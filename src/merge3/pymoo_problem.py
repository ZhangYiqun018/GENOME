import os
import random
import shutil
import numpy as np
from typing import List, Optional, Sequence

from pymoo.core.problem import Problem

from src.genome.individual import Individual
from src.merge3.merger import LoRAMerger
from loguru import logger


class Merge3PymooProblem(Problem):
    """pymoo-compatible wrapper around Merge3 merging/evaluation."""

    def __init__(
        self,
        method,
        merger: LoRAMerger,
        pools: Sequence[str],
        parent_sample_size: int,
        genotype_dimension: int,
        variable_bounds: Sequence[float],
        seed: int,
        save_intermediate: bool,
    ) -> None:
        if parent_sample_size < 2:
            raise ValueError("Merge3 requires at least two parent adapters.")
        if len(pools) < parent_sample_size:
            raise ValueError("Not enough pools to sample parents for Merge3.")
        if len(variable_bounds) != 2:
            raise ValueError("variable_bounds must contain (low, high).")

        self.method = method
        self.merger = merger
        self.pools: List[str] = list(pools)
        self.parent_sample_size = parent_sample_size
        self.rng = random.Random(seed)
        self.save_intermediate = save_intermediate
        self.best_score: float = float("-inf")
        self.best_individual: Optional[Individual] = None
        self.best_task_scores: Optional[dict] = None
        self.best_path: Optional[str] = None

        lo, hi = variable_bounds
        super().__init__(
            n_var=genotype_dimension,
            n_obj=1,
            n_eq_constr=0,
            n_ieq_constr=0,
            xl=np.full(genotype_dimension, lo),
            xu=np.full(genotype_dimension, hi),
            elementwise=True,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        genotype = [float(val) for val in np.asarray(x).tolist()]
        parent_paths = self.rng.sample(self.pools, self.parent_sample_size)
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

        scores = self.method.evaluate(individuals=[individual])
        if not scores or individual.id not in scores:
            raise RuntimeError(
                f"No evaluation results returned for individual {individual.id}; "
                "check earlier evaluation logs for errors."
            )

        weighted_score = scores[individual.id]["weighted_score"]
        out["F"] = [-weighted_score]

        if weighted_score > self.best_score:
            if (
                not self.save_intermediate
                and self.best_path
                and os.path.isdir(self.best_path)
                and self.best_path != out_dir
            ):
                logger.debug(f"Deleting previous best adapter at {self.best_path}")
                shutil.rmtree(self.best_path, ignore_errors=True)
            self.best_score = weighted_score
            self.best_individual = individual
            self.best_task_scores = scores[individual.id]["task_scores"]
            self.best_path = out_dir
        elif not self.save_intermediate:
            logger.debug(f"Deleting non-best adapter at {out_dir}")
            shutil.rmtree(out_dir, ignore_errors=True)
