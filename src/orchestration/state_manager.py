from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from loguru import logger


class StateManager:
    """Manage optimization state persistence and reporting."""

    def __init__(self, workspace: str):
        self.workspace = workspace
        self._state: Dict[str, Any] = {}

    @property
    def state(self) -> Dict[str, Any]:
        return self._state

    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise TypeError("State must be a dictionary")
        self._state = value

    def load(self) -> None:
        """Load the persisted state if available."""
        state_path = os.path.join(self.workspace, "state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                self._state = json.load(f)
            logger.info(f"Loaded state from {state_path}")
        else:
            self._state = {}

    def save(self) -> None:
        """Persist the current state to disk."""
        os.makedirs(self.workspace, exist_ok=True)
        state_path = os.path.join(self.workspace, "state.json")
        with open(state_path, "w") as f:
            json.dump(self._state, indent=4, ensure_ascii=False, fp=f)
        logger.info(f"State saved to {state_path}")

    def update_step(
        self,
        step: int,
        *,
        time: float,
        individuals: Sequence[Any],
        global_state: Dict[str, Any],
        weighted_scores: Optional[Dict[str, Dict[str, Any]]] = None,
        tasks: Optional[Iterable[str]] = None,
    ) -> None:
        """Update optimization metrics for a specific step."""
        if tasks is None and weighted_scores:
            tasks = list(next(iter(weighted_scores.values()))["task_scores"].keys())
        tasks = list(tasks or [])

        fitness_scores = [getattr(individual, "fitness_score", 0.0) for individual in individuals]
        average_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0

        step_state: Dict[str, Any] = {
            "global_max_fitness_path": global_state.get("max_path", ""),
            "global_max_fitness_score": global_state.get("max_score", 0.0),
            "global_min_fitness_path": global_state.get("min_path", ""),
            "global_min_fitness_score": global_state.get("min_score", 0.0),
            "average_fitness_score": average_fitness,
            "consume_time": time,
        }

        if weighted_scores:
            task_stats: Dict[str, Dict[str, float]] = {
                task: {"max": float("-inf"), "min": float("inf"), "sum": 0.0}
                for task in tasks
            }
            weighted_stats = {"max": float("-inf"), "min": float("inf"), "sum": 0.0}

            for individual_data in weighted_scores.values():
                weighted_score = individual_data["weighted_score"]
                weighted_stats["max"] = max(weighted_stats["max"], weighted_score)
                weighted_stats["min"] = min(weighted_stats["min"], weighted_score)
                weighted_stats["sum"] += weighted_score

                for task, score in individual_data["task_scores"].items():
                    task_stats.setdefault(task, {"max": float("-inf"), "min": float("inf"), "sum": 0.0})
                    task_stats[task]["max"] = max(task_stats[task]["max"], score)
                    task_stats[task]["min"] = min(task_stats[task]["min"], score)
                    task_stats[task]["sum"] += score

            n_individuals = max(len(weighted_scores), 1)
            step_state["weighted_scores"] = {
                "max": weighted_stats["max"],
                "min": weighted_stats["min"],
                "avg": weighted_stats["sum"] / n_individuals,
            }
            step_state["task_scores"] = {
                task: {
                    "max": stats["max"],
                    "min": stats["min"],
                    "avg": stats["sum"] / n_individuals,
                }
                for task, stats in task_stats.items()
            }
        else:
            step_state["all_fitness_score"] = fitness_scores

        self._state[f"step_{step}"] = step_state

    def report_step(self, step: int) -> None:
        key = f"step_{step}"
        if key not in self._state:
            logger.warning(f"No state recorded for step {step}")
            return

        state = self._state[key]
        logger.info(
            "Step: {step}, Global max: {gmax:.4f}, Global min: {gmin:.4f}, Average fitness score: {avg:.4f}".format(
                step=step,
                gmax=state.get("global_max_fitness_score", 0.0),
                gmin=state.get("global_min_fitness_score", 0.0),
                avg=state.get("average_fitness_score", 0.0),
            )
        )

