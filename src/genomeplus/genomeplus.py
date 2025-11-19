import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import json
import shutil
import numpy as np
from loguru import logger
from src.genomeplus.config import GenomePlusConfig
from src.utils import load_lora_weight
from src.genomeplus.individual import Individual
from src.base.base_method import BaseMethod
import math


class GenomePlus(BaseMethod):    
    def __init__(self, config: GenomePlusConfig) -> None:
        """Initialize the hybrid optimizer.
        
        Args:
            config (GenomePlusConfig): Configuration object containing all hyperparameters
                                 and settings for the optimization process.
        """
        self.cross_method = config.cross_method
        super().__init__(config)
        config.validate()

        # TODO
        # N_init, N_max
        self.N_init = config.N_init
        self.N_max = config.N_max
        self.do_init = config.do_init
        self.max_iters = config.max_iter

        # hyperparameters
        ## PSO
        self.lambda_step = config.lambda_step
        self.phi_lambda = config.phi_lambda
        self.phi_inertia = config.phi_inertia
        self.phi_cognitive = config.phi_cognitive
        self.phi_social = config.phi_social
        self.phi_repel = config.phi_repel
        self.phi = [self.phi_inertia, self.phi_cognitive, self.phi_social, self.phi_repel]
        
        self.unable_random = config.unable_random
        
        ## GA
        self.cross_rate = config.cross_rate
        self.individual_mutation_rate = config.individual_mutation_rate
        self.gene_mutation_rate = config.gene_mutation_rate
        self.sigma = config.sigma
        self.elite_percent = config.elite_percent
        self.method = config.method
        self.selection_method = config.selection_method
        # init population
        self.individuals = []
        
        self.random_r()
        self.compute_C()

    def print_config(self) -> None:
        """Print the configuration of the hybrid optimizer."""
        logger.info(f"GENOME+ config:")
        logger.info(f"Tasks = {self.tasks}, Task weights = {self.task_weights}")
        logger.info(f"Cross method = {self.cross_method.value}")
        logger.info(f"Combine method = {self.combine_method.value}")
        logger.info(f"Init population size: {self.N_init}, max population size: {self.N_max}")
        logger.info(f"PSO hyperparameters: lambda_step: {self.lambda_step}, phi_lambda: {self.phi_lambda}, phi_inertia: {self.phi_inertia}, phi_cognitive: {self.phi_cognitive}, phi_social: {self.phi_social}, phi_repel: {self.phi_repel}")
        logger.info(f"GA hyperparameters: cross_rate: {self.cross_rate}, individual_mutation_rate: {self.individual_mutation_rate}, gene_mutation_rate: {self.gene_mutation_rate}, sigma: {self.sigma}, elite_percent: {self.elite_percent}")
        logger.info(f"Early stop: {self.early_stop}, early_stop_iter: {self.early_stop_iter}")
        
    # PSO parameters
    def random_r(self) -> None:
        """Generate random coefficients for PSO velocity update."""
        if self.unable_random:
            self.r = [1, 1, 1, 1]
        else:
            r = random.Random(self.seed)
            self.r = [r.random() for _ in range(4)]
    
    def compute_C(self) -> None:
        """Compute the normalization factor C for PSO velocity update."""
        self.C = sum(
            [self.r[i] * self.phi[i] for i in range(4)]
        )
    
    def update_lambda(self) -> None:
        """Update lambda step size using decay factor."""
        self.lambda_step *= self.phi_lambda
            
    def initialize(self) -> None:
        """Initialize the population for both PSO and GA optimization.
        
        This method:
        1. Creates initial population by merging expert LoRAs
        2. Initializes velocities for PSO particles
        3. Evaluates initial population fitness
        """
        start_time = time.time()
        if self.do_init:
            logger.info(f"Initializing population, init number: {self.N_init} ...")
            expert_pairs = self.generate_pair_sequences(pools=self.pools, n_samples=self.N_init, seed=self.seed)
            for expert_pair in expert_pairs:
                individual_weight = self.merge_lora_weights(
                    lora_state_dicts=[load_lora_weight(expert) for expert in expert_pair],
                    weights=[],
                    method=self.combine_method,
                    density=0.7,
                    majority_sign_method="total"
                )
                individual_id = uuid.uuid4().hex
                individual = Individual(
                    id = individual_id,
                    x = individual_weight,
                    weight_path=os.path.join(self.workspace, f"individual_{individual_id}"),
                    parent=expert_pair,
                    lora_config_path=self.pools[0],
                    model_name_or_path=self.model_name_or_path,
                )
                individual.save_individual(save_path=individual.weight_path)
                self.individuals.append(individual)
        else:
            logger.warning("do not init population, use default experts as initial population, N_init is 10!")
            for pool in self.pools:
                individual_id = uuid.uuid4().hex
                individual_weight = load_lora_weight(pool)
                individual = Individual(
                    id = individual_id,
                    x = individual_weight,
                    weight_path=os.path.join(self.workspace, f"individual_{individual_id}"),
                    parent=[],
                    lora_config_path=self.pools[0],
                    model_name_or_path=self.model_name_or_path,
                )
                individual.save_individual(save_path=individual.weight_path)
                self.individuals.append(individual)
        weighted_scores = self.evaluate(individuals=self.individuals, split="valid")
        for individual_id, result in weighted_scores.items():
            self.update_global(
                id=individual_id,
                fitness_score=result['weighted_score'],
                path=result['path'],
                task_scores=result['task_scores']
            )
        
        # init every individual's velocity
        logger.info(f"Initializing velocities ...")
        for individual in self.individuals:
            if individual.v is None:
                individual.init_velocity(
                    lora=random.choice(self.individuals).x
                )
        
        end_time = time.time()
        logger.info(f"Init time: {(end_time - start_time):.2f} seconds.")
        
        self.state_manager.update_step(
            step=0,
            time=end_time - start_time,
            individuals=self.individuals,
            weighted_scores=weighted_scores,
            tasks=self.tasks,
            global_state=self.get_global_state_snapshot(),
        )
        self.state_manager.save()
        self.state_manager.report_step(step=0)

    def crossover(self, method: str, parent_size: int = 2) -> None:
        """Perform crossover operation for GA optimization.
        
        Args:
            step (int): Current optimization step
            method (str): Crossover method to use (e.g. 'roulette')
            parent_size (int, optional): The number of parents to crossover
        """
        logger.info(f"Start crossover, method: {method} ...")
        if len(self.individuals) < parent_size:
            logger.warning(f"Population size is less than {parent_size}, no need to crossover.")
            return
        
        num_pairs = len(self.individuals) // parent_size
        if method == "roulette":
            fitness_scores = [i.fitness_score for i in self.individuals]
            total_fitness = sum(fitness_scores)
            prob = [i / total_fitness for i in fitness_scores]
            pairs = []
            for _ in range(num_pairs):
                if random.random() <= self.cross_rate:
                    parents = np.random.choice(
                        self.individuals, size=parent_size, p=prob, 
                        replace=False,
                    )
                    pairs.append(parents)
            
        else:
            raise ValueError(f"Invalid crossover method: {method}")
        
        for pair in pairs:
            individual_id = uuid.uuid4().hex
            individual_weight = self.merge_lora_weights(
                lora_state_dicts=[p.x for p in pair],
                weights=[p.fitness_score for p in pair],
                method=self.cross_method,
                density=0.7,
                majority_sign_method="total"
            )
            individual: Individual = Individual(
                id=individual_id,
                x=individual_weight,
                weight_path=os.path.join(self.workspace, f"individual_{individual_id}"),
                parent=[pair[0].weight_path, pair[1].weight_path],
                lora_config_path=self.pools[0],
                model_name_or_path=self.model_name_or_path,
            )
            individual.save_individual(save_path=individual.weight_path)
            self.individuals.append(individual)
        
        logger.info(f"Crossover completed, current population size: {len(self.individuals)}")
        
    def mutation(self) -> None:
        """Perform mutation operation for GA optimization.
        
        Applies mutation to individuals based on mutation rates and adds
        mutated individuals to population.
        """
        logger.info(f"Start mutation ...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    individual.mutation,
                    individual_mutation_rate=self.individual_mutation_rate,
                    gene_mutation_rate=self.gene_mutation_rate,
                    sigma=self.sigma
                ) for individual in self.individuals
            ]
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    self.individuals.append(result)
        
        for individual in self.individuals:
            if individual.v is None:
                individual.init_velocity(
                    lora=random.choice(self.individuals).x
                )

        logger.info(f"Mutation completed, current population size: {len(self.individuals)}")
    
    def selection(self, method: str, number: int) -> List[Individual]:
        """Select individuals for next generation.

        Args:
            method (str): Selection method to use (e.g. 'roulette', 'random')
            number (int): The number of individuals to select
        Returns:
            List[Individual]: Selected individuals
        """
        if len(self.individuals) <= number:
            logger.info(f"Population size (GA) is equal to N_GA (N={number}), no need to select.")
            return self.individuals
        
        logger.info(f"Start selection, method: {method} ...")
        all_individual_paths = {ind.weight_path for ind in self.individuals}
        
        sorted_individuals = sorted(
            self.individuals,
            key=lambda x: x.fitness_score, reverse=True
        )
        elite_number = round(len(sorted_individuals) * self.elite_percent)
        elites = sorted_individuals[:elite_number]
        remaining_individuals = sorted_individuals[elite_number:]
        
        remaining_size = number - elite_number
        
        if remaining_size <= 0:
            return elites
        
        if method == "roulette":
            fitness_scores = [ind.fitness_score for ind in remaining_individuals]
            prob = [score/sum(fitness_scores) for score in fitness_scores]
            selected_remaining = np.random.choice(
                remaining_individuals,
                size = remaining_size,
                p = prob,
                replace=False
            )
            selected_individuals = np.concatenate(
                [elites, selected_remaining]
            )
        elif method == "random":
            logger.info(f"random selection ...")
            selected_individuals = np.random.choice(
                sorted_individuals,
                size=remaining_size,
                replace=False,
            )
            selected_individuals = np.concatenate(
                [elites, selected_individuals]
            )
        else:
            raise ValueError(f"Invalid selection method: {method}")

        selected_paths = {ind.weight_path for ind in selected_individuals}
        paths_to_remove = all_individual_paths - selected_paths
        for path in paths_to_remove:
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    logger.info(f"Removed unselected individual: {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {str(e)}")
        
        return selected_individuals.tolist()
    
    def _step(self, step: int) -> None:
        """Execute one step of PSO optimization.
        
        Args:
            global_step (int): Current global iteration number
            
        Process:
            1. Update particle velocities using standard PSO formula
            2. Update particle positions (weights)
            3. Evaluate fitness of new individuals
            4. Crossover
            5. Mutation
            6. Evaluate fitness of new individuals
            7. Select individuals for next stage
        """
        start_time = time.time()
        logger.info(f"Start step {step} ...")
        logger.info(f"Updating velocities & weights ...")
        # 1-3, PSO
        self.update_velocity()
        self.update_weight()
        weighted_scores = self.evaluate(individuals=self.individuals, split="valid")
        for individual_id, result in weighted_scores.items():
            self.update_global(
                id=individual_id,
                fitness_score=result['weighted_score'],
                path=result['path'],
                task_scores=result['task_scores']
            )
        # 4-7, GA
        self.crossover(method=self.method, parent_size=2)
        self.mutation()
        weighted_scores = self.evaluate(individuals=self.individuals, split="valid")
        for individual_id, result in weighted_scores.items():
            self.update_global(
                id=individual_id,
                fitness_score=result['weighted_score'],
                path=result['path'],
                task_scores=result['task_scores']
            )
        if len(self.individuals) > self.N_max:
            self.individuals = self.selection(method=self.selection_method, number=self.N_max)
        
        end_time = time.time()
        logger.info(f"Step {step} consume time: {(end_time - start_time):.2f} seconds.")
        self.state_manager.update_step(
            step=step,
            time=end_time - start_time,
            individuals=self.individuals,
            weighted_scores=weighted_scores,
            tasks=self.tasks,
            global_state=self.get_global_state_snapshot(),
        )
        self.state_manager.save()
        self.state_manager.report_step(step=step)
    
    def search(self) -> None:
        """Start Hybrid (PSO-GA) search process.
        
        Process:
            1. Initialize population
            2. Start global iteration
            3. Evaluate final model on test set using ensemble inference
        """
        start_time = time.time()
        logger.info("Start Hybrid (PSO-GA) search ...")
        
        self.print_config()
        self.initialize()
        
        for step in range(1, self.max_iters+1):
            self._step(step)
        
        # test performance
        self.ensemble_test(individuals=self.individuals, split="test")
        
        try:
            self.save_final_state(individuals=self.individuals, time=time.time() - start_time)
        except Exception as e:
            self.state_manager.save()
            logger.error(f"Save final state failed: {e}")

        if self.plot_enabled:
            try:
                self.generate_plots()
            except Exception as e:
                logger.error(f"Error generating plots: {e}")
        
    def update_velocity(self) -> None:
        """Update velocities for all PSO particles."""
        for individual in self.individuals:
            individual.update_velocity(
                global_max_weight=self.global_max_fitness_weight,
                global_min_weight=self.global_min_fitness_weight,
                r = self.r,
                phi = [self.phi_inertia, self.phi_cognitive, self.phi_social, self.phi_repel],
                C = self.C,
            )
    
    def update_weight(self) -> None:
        """Update weights (positions) for all PSO particles."""
        for individual in self.individuals:
            self.update_lambda()
            individual.update_weight(
                _lambda=self.lambda_step
            )
            individual.save_individual(
                save_path=individual.weight_path
            )