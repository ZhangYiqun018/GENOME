import json
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

from loguru import logger
from openai import OpenAI

from src.swarms.config import PSOConfig
from src.swarms.particle import Particle
from src.utils import load_lora_weight
from src.base.base_method import BaseMethod


class PSO(BaseMethod):
    def __init__(self, config: PSOConfig) -> None:
        super().__init__(config)
        self.config = config
        self.config.validate()
        
        self.config.save(path=self.workspace)
        
        self.global_patience_counter = 0
        self.patience_flag = True
        
        self.phi = [self.config.phi_inertia, self.config.phi_cognitive, self.config.phi_social, self.config.phi_repel]
        
        # max particle number
        self._lambda = self.config.lambda_step
        
        self.unable_random = self.config.unable_random
        
        self.random_r()
        self.compute_C()
        
        self.lora_config = self.config.pools[0]
        
    def random_r(self)-> None:
        if self.unable_random:
            self.r = [1, 1, 1, 1]
        else:
            r = random.Random(self.seed)
            self.r = [r.random() for _ in range(4)]

    def compute_C(self) -> None:
        self.C = sum([self.r[i]*self.phi[i] for i in range(4)])        
    
    def update_lambda(self) -> None:
        self._lambda *= self.config.phi_lambda
    
    def print_config(self) -> None:
        logger.info(
            f"Config: N={self.config.N}, max_iter={self.config.max_iter}, "
            f"tasks={self.tasks}, task_weights={self.task_weights}, "
            f"max_workers={self.config.max_workers}\n"
            f"Combine method: {self.combine_method}\n"
            f"Weights (φ): inertia={self.config.phi_inertia:.3f}, "
            f"cognitive={self.config.phi_cognitive:.3f}, "
            f"social={self.config.phi_social:.3f}, "
            f"repel={self.config.phi_repel:.3f}, "
            f"lambda={self._lambda:.3f}, "
            f"phi_lambda={self.config.phi_lambda:.3f}\n"
            f"Random (r): unable={self.unable_random}, "
            f"inertia={self.r[0]:.3f}, "
            f"cognitive={self.r[1]:.3f}, "
            f"social={self.r[2]:.3f}, "
            f"repel={self.r[3]:.3f}, "
        )
    
    def initialize(self) -> None:
        logger.info("Initializing individuals ...")
        
        start_time = time.time()
        self.individuals = []
        expert_pairs = self.generate_pair_sequences(pools=self.pools, n_samples=self.config.N, seed=self.seed)
        
        for expert_pair in expert_pairs:
            particle_weight = self.merge_lora_weights(
                lora_state_dicts=[load_lora_weight(expert) for expert in expert_pair],
                weights=[],
                method=self.combine_method,
                density=0.7,
                majority_sign_method="total"
            )
            particle_id = uuid.uuid4().hex
            p = Particle(
                id=particle_id,
                x=particle_weight,
                parent=expert_pair,
                weight_path=os.path.join(self.workspace, f"particle_{particle_id}"),
                lora_config_path=self.lora_config,
                model_name_or_path=self.model_name_or_path,
            )
            p.save_particle(save_path=p.weight_path)
            self.individuals.append(p)
            
        weighted_scores = self.evaluate(individuals=self.individuals, split="valid")
        for individual_id, result in weighted_scores.items():
            self.update_global(
                id=individual_id,
                fitness_score=result['weighted_score'],
                path=result['path'],
                task_scores=result['task_scores']
            )
        # Initialize velocity
        logger.info("Initializing velocity ...")
        for p in self.individuals:
            p.init_velocity(
                lora=random.choice(self.individuals).x
            )
    
        end_time = time.time()
        
        logger.info(f"Initialization finished in {end_time-start_time:.2f} seconds.")
        
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
    
    def velocity_update(self) -> None:
        for p in self.individuals:
            assert hasattr(self, "global_max_fitness_weight") and hasattr(self, "global_min_fitness_weight"), "Global max and min fitness weight must be set."
            p.update_velocity(
                global_max_weight=self.global_max_fitness_weight,
                global_min_weight=self.global_min_fitness_weight,
                r = self.r,
                phi = self.phi,
                C = self.C
            )
    
    def weight_update(self, step: int) -> None:
        for p in self.individuals:
            self.update_lambda()
            p.update_weight(_lambda=self._lambda)
            save_path = os.path.join(self.workspace, f"particle_{p.id}")
            p.save_particle(save_path=save_path)
            p.evaluated = {task: False for task in self.tasks}
    
    def single_search(self, step: int) -> None:
        """Preform a single search step."""
        start_time = time.time()
        
        logger.info(f"Start searching for {step} steps ...")
        logger.info("Update velocity & weight ...")
        self.velocity_update()
        self.weight_update(step=step)
        
        logger.info("Evaluating update position ...")
        weighted_scores = self.evaluate(individuals=self.individuals, split="valid")
        for individual_id, result in weighted_scores.items():
            self.update_global(
                id=individual_id,
                fitness_score=result['weighted_score'],
                path=result['path'],
                task_scores=result['task_scores']
            )
        
        end_time = time.time()
        logger.info(f"Step {step} finished in {end_time-start_time:.2f} seconds.")
        
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
        
    def search(self):
        start_time = time.time()
        self.print_config()
        self.initialize()
        for step in range(1, self.config.max_iter+1):
            self.single_search(step=step)
            
            if self.patience_flag == True:
                self.global_patience_counter += 1
                if self.global_patience_counter >= self.early_stop and self.config.early_stop:
                    logger.info("Early stop!")
                    break
            else:
                self.global_patience_counter = 0
                self.patience_flag = True
        
        # get test performance
        self.ensemble_test(individuals=self.individuals, split="test")
        
        end_time = time.time()
        try:
            self.save_final_state(individuals=self.individuals, time=end_time-start_time)
        except Exception as e:
            self.state_manager.save()
            logger.error(f"Error saving final state: {e}")
            
        if self.plot_enabled:
            try:
                self.generate_plots()
            except Exception as e:
                logger.error(f"Error generating plots: {e}")