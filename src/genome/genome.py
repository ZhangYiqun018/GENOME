import time
from typing import List

from loguru import logger
from src.genome.config import GenomeConfig
from src.genome.individual import Individual
from src.utils import load_lora_weight
from src.base.base_method import BaseMethod
from src.orchestration.optimizer import GenomeOptimizer

class Genome(BaseMethod):
    def __init__(self, config: GenomeConfig):
        """
        Initialize the Genome.
        """
        
        # order is important.
        self.cross_method = config.cross_method
        super().__init__(config)
        config.validate()
        
        self.N = config.N
        self.epochs = config.max_iter

        # hyper params
        self.cross_rate = config.cross_rate
        self.individual_mutation_rate = config.individual_mutation_rate
        self.gene_mutation_rate = config.gene_mutation_rate
        self.sigma = config.sigma
        self.elite_percent = config.elite_percent
        self.elite_number = int(self.elite_percent * self.N)

        self.method = config.method
        self.individuals: List[Individual] = []

        self.optimizer = GenomeOptimizer(
            config=config,
            evaluator=self,
            inference_client=self,
            state_manager=self.state_manager,
            workspace=self.workspace,
            pools=self.pools,
            combine_method=self.combine_method,
            model_name_or_path=self.model_name_or_path,
            seed=self.seed,
            max_workers=self.max_workers,
            merge_lora_weights=self.merge_lora_weights,
            load_lora_weight_fn=load_lora_weight,
            generate_pair_sequences=self.generate_pair_sequences,
            update_global=self.update_global,
            get_global_state=self.get_global_state_snapshot,
            get_global_max_fitness_score=lambda: self.global_max_fitness_score,
        )

    def initialize(self) -> None:
        self.optimizer.initialize_population(self.individuals, self.tasks)

    def selection(self, method: str) -> List[Individual]:
        self.individuals = self.optimizer.selection(self.individuals, method)
        return self.individuals

    def crossover(self, step: int, method: str):
        self.optimizer.crossover(self.individuals, step=step, method=method)

    def mutation(self):
        self.optimizer.mutation(self.individuals)

    def _step(self, step: int, method: str):
        self.optimizer.step(
            self.individuals,
            step=step,
            tasks=self.tasks,
            crossover_method=method,
            selection_method="tournament",
        )
    
    def print_config(self):
        logger.info(f"GA config: ")
        logger.info(
            f"Tasks = {self.tasks}, Task weights = {self.task_weights}, "
            f"Seed = {self.seed}, N = {self.N}, Epochs = {self.epochs}, "
            f"Combine method = {self.combine_method}, "
            f"Cross method = {self.cross_method}, "
            f"cross rate = {self.cross_rate}, mutation rate = {self.individual_mutation_rate}, "
            f"gene mutation rate = {self.gene_mutation_rate}, sigma = {self.sigma}\n"
            f"Early Stop = {self.early_stop}, Early Stop Iter = {self.early_stop_iter}\n"
        )
    
    def search(self):
        start_time = time.time()
        logger.info("Start GA search.")
        self.print_config()

        method = self.method
        self.initialize()
        for i in range(1, self.epochs+1):
            self._step(step=i, method=method)

            if self.patience_flag == True:
                self.global_patience_counter += 1
                if self.global_patience_counter > self.early_stop_iter and self.early_stop:
                    logger.info("Early stop triggered.")
                    break
            else:
                self.global_patience_counter = 0
                self.patience_flag = True
            
        # test performance
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