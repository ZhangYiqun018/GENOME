from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from loguru import logger

from src.genome.individual import Individual
from src.orchestration.state_manager import StateManager


class GenomeOptimizer:
    """Encapsulate the core evolutionary operations for GENOME."""

    def __init__(
        self,
        *,
        config: Any,
        evaluator: Any,
        inference_client: Any,
        state_manager: StateManager,
        workspace: str,
        pools: Sequence[str],
        combine_method: Any,
        model_name_or_path: str,
        seed: int,
        max_workers: int,
        merge_lora_weights: Callable[..., Dict[str, Any]],
        load_lora_weight_fn: Callable[[str], Dict[str, Any]],
        generate_pair_sequences: Callable[[Sequence[str], int, Optional[int]], Iterable[Sequence[str]]],
        update_global: Callable[[str, float, str, Dict[str, float]], None],
        get_global_state: Callable[[], Dict[str, Any]],
        get_global_max_fitness_score: Callable[[], float],
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.inference_client = inference_client
        self.state_manager = state_manager
        self.workspace = workspace
        self.pools = list(pools)
        self.combine_method = combine_method
        self.model_name_or_path = model_name_or_path
        self.seed = seed
        self.max_workers = max_workers
        self.merge_lora_weights = merge_lora_weights
        self.load_lora_weight_fn = load_lora_weight_fn
        self.generate_pair_sequences = generate_pair_sequences
        self.update_global = update_global
        self.get_global_state = get_global_state
        self.get_global_max_fitness_score = get_global_max_fitness_score

        self.N = config.N
        self.cross_method = config.cross_method
        self.cross_rate = config.cross_rate
        self.individual_mutation_rate = config.individual_mutation_rate
        self.gene_mutation_rate = config.gene_mutation_rate
        self.sigma = config.sigma
        self.elite_percent = config.elite_percent
        self.elite_number = int(self.elite_percent * self.N)
        self.method = config.method

    def initialize_population(self, individuals: List[Individual], tasks: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        logger.info("Initializing population...")
        start_time = time.time()
        individuals.clear()

        expert_pairs = self.generate_pair_sequences(pools=self.pools, n_samples=self.N, seed=self.seed)
        for expert_pair in expert_pairs:
            individual_id = uuid.uuid4().hex
            individual_weight = self.merge_lora_weights(
                lora_state_dicts=[
                    self.load_lora_weight_fn(expert_pair[0]),
                    self.load_lora_weight_fn(expert_pair[1]),
                ],
                weights=[],
                method=self.combine_method,
                density=0.7,
                majority_sign_method="total",
            )
            individual = Individual(
                id=individual_id,
                x=individual_weight,
                weight_path=os.path.join(self.workspace, f"individual_{individual_id}"),
                parent=expert_pair,
                lora_config_path=self.pools[0],
                model_name_or_path=self.model_name_or_path,
            )
            individual.save_individual(save_path=individual.weight_path)
            individuals.append(individual)

        weighted_scores = self.evaluator.evaluate(individuals=individuals, split="valid")
        for individual_id, result in weighted_scores.items():
            self.update_global(
                id=individual_id,
                fitness_score=result["weighted_score"],
                path=result["path"],
                task_scores=result["task_scores"],
            )

        elapsed = time.time() - start_time
        logger.info(f"Init time: {elapsed:.2f} seconds.")

        self.state_manager.update_step(
            step=0,
            time=elapsed,
            individuals=individuals,
            weighted_scores=weighted_scores,
            tasks=tasks,
            global_state=self.get_global_state(),
        )
        self.state_manager.save()
        self.state_manager.report_step(step=0)
        return weighted_scores

    def selection(self, individuals: List[Individual], method: str) -> List[Individual]:
        if len(individuals) == self.N:
            logger.info(f"Population size is equal to N (N={self.N}).")
            return individuals

        logger.info(f"Start selection, method: {method}")
        all_individual_paths = {ind.weight_path for ind in individuals}

        sorted_individuals = sorted(individuals, key=lambda x: x.fitness_score, reverse=True)
        elites = sorted_individuals[: self.elite_number]
        remaining_individuals = sorted_individuals[self.elite_number :]
        remaining_size = self.N - self.elite_number

        if method == "roulette":
            fitness_scores = [i.fitness_score for i in remaining_individuals]
            prob = [score / sum(fitness_scores) for score in fitness_scores]
            selected_remaining = np.random.choice(
                remaining_individuals,
                size=remaining_size,
                p=prob,
                replace=False,
            )
            selected_individuals = np.concatenate([elites, selected_remaining])
        elif method == "tournament":
            tournament_size = 3
            selected_remaining = []
            available_individuals = remaining_individuals.copy()
            while len(selected_remaining) < remaining_size and available_individuals:
                current_size = min(tournament_size, len(available_individuals))
                tournament_candidates = np.random.choice(
                    available_individuals,
                    size=current_size,
                    replace=False,
                )
                winner = max(tournament_candidates, key=lambda x: x.fitness_score)
                selected_remaining.append(winner)
                available_individuals.remove(winner)
        elif method == "rank":
            ranks = range(1, len(remaining_individuals) + 1)
            prob = [rank / sum(ranks) for rank in ranks]
            selected_remaining = np.random.choice(
                remaining_individuals,
                size=remaining_size,
                replace=False,
                p=prob,
            )
        elif method == "elite":
            selected_remaining = sorted_individuals[self.elite_number : self.N]
        elif method == "random":
            selected_remaining = np.random.choice(
                remaining_individuals,
                size=remaining_size,
                replace=False,
            )
        else:
            raise ValueError(f"Unknown selection method: {method}")

        selected_individuals = np.concatenate([elites, selected_remaining])
        selected_list = selected_individuals.tolist()

        selected_paths = {ind.weight_path for ind in selected_list}
        paths_to_remove = all_individual_paths - selected_paths
        for path in paths_to_remove:
            if os.path.exists(path):
                try:
                    import shutil

                    shutil.rmtree(path)
                    logger.info(f"Removed unselected individual: {path}")
                except Exception as exc:
                    logger.warning(f"Failed to remove {path}: {exc}")

        return selected_list

    def crossover(self, individuals: List[Individual], step: int, method: str) -> None:
        logger.info(f"Start Crossover, method: {method}")
        num_pairs = len(individuals) // 2
        pairs = []

        if method == "roulette":
            fitness_scores = [i.fitness_score for i in individuals]
            total_fitness = sum(fitness_scores)
            prob = [score / total_fitness for score in fitness_scores]

            for _ in range(num_pairs):
                if np.random.random() <= self.cross_rate:
                    parents = np.random.choice(individuals, size=2, p=prob, replace=False)
                    pairs.append((parents[0], parents[1]))
        elif method == "random":
            for _ in range(num_pairs):
                parents = np.random.choice(individuals, size=2, replace=False)
                pairs.append((parents[0], parents[1]))
        else:
            raise ValueError(f"Invalid method: {method}")

        for parent_a, parent_b in pairs:
            individual_id = uuid.uuid4().hex
            individual_weight = self.merge_lora_weights(
                lora_state_dicts=[parent_a.x, parent_b.x],
                weights=[parent_a.fitness_score, parent_b.fitness_score],
                method=self.cross_method,
                density=0.7,
                majority_sign_method="total",
                alpha=1.2,
            )
            individual = Individual(
                id=individual_id,
                x=individual_weight,
                weight_path=os.path.join(self.workspace, f"individual_{individual_id}"),
                parent=(parent_a, parent_b),
                lora_config_path=self.pools[0],
                model_name_or_path=self.model_name_or_path,
            )
            individual.save_individual(save_path=individual.weight_path)
            individuals.append(individual)

        logger.info(f"Crossover completed, current population size: {len(individuals)}")

    def mutation(self, individuals: List[Individual]) -> None:
        logger.info("Start mutation...")
        best_fitness_score = self.get_global_max_fitness_score()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    individual.mutation,
                    individual_mutation_rate=self.individual_mutation_rate,
                    gene_mutation_rate=self.gene_mutation_rate,
                    sigma=self.sigma,
                    best_fitness_score=best_fitness_score,
                )
                for individual in individuals
            ]

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    individuals.append(result)

        logger.info(f"Mutation completed, current population size: {len(individuals)}")

    def step(
        self,
        individuals: List[Individual],
        *,
        step: int,
        tasks: Sequence[str],
        crossover_method: Optional[str] = None,
        selection_method: str = "tournament",
    ) -> Dict[str, Dict[str, Any]]:
        start_time = time.time()
        crossover_method = crossover_method or self.method

        self.crossover(individuals, step=step, method=crossover_method)
        self.mutation(individuals)

        weighted_scores = self.evaluator.evaluate(individuals=individuals, split="valid")
        for individual_id, result in weighted_scores.items():
            self.update_global(
                id=individual_id,
                fitness_score=result["weighted_score"],
                path=result["path"],
                task_scores=result["task_scores"],
            )

        updated_population = self.selection(individuals, method=selection_method)
        individuals.clear()
        individuals.extend(updated_population)

        elapsed = time.time() - start_time
        logger.info(f"Step {step} takes {elapsed:.2f} seconds.")

        self.state_manager.update_step(
            step=step,
            time=elapsed,
            individuals=individuals,
            weighted_scores=weighted_scores,
            tasks=tasks,
            global_state=self.get_global_state(),
        )
        self.state_manager.save()
        self.state_manager.report_step(step=step)
        return weighted_scores

