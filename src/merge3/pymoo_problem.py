import numpy as np
from typing import List, Optional, Sequence

from pymoo.core.problem import Problem

from src.genome.individual import Individual
from src.merge3.merger import LoRAMerger


class Merge3PymooProblem(Problem):
    """pymoo-compatible wrapper around Merge3 merging/evaluation."""

    def __init__(
        self,
        method,
        merger: LoRAMerger,
        parent_paths: Sequence[str],
        genotype_dimension: int,
        variable_bounds: Sequence[float],
    ) -> None:
        if len(parent_paths) < 2:
            raise ValueError("Merge3 requires at least two parent adapters.")
        if len(variable_bounds) != 2:
            raise ValueError("variable_bounds must contain (low, high).")

        self.method = method
        self.merger = merger
        self.parent_paths: List[str] = list(parent_paths)
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
        out_dir, merged_state = self.merger.materialize(genotype, self.parent_paths)

        individual = Individual(
            id=out_dir.split("_")[-1],
            x=merged_state,
            parent=list(self.parent_paths),
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
            self.best_score = weighted_score
            self.best_individual = individual
            self.best_task_scores = scores[individual.id]["task_scores"]
            self.best_path = scores[individual.id]["path"]
