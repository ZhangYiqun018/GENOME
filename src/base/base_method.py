import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Literal
import json
import os
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
import torch
from src.evaluate.eval import CombineMethod
from src.utils import load_lora_weight
from src.base.merge_utils import process_ties, process_dare_linear, process_dare_ties, process_linear, process_blxalpha, process_random
from openai import OpenAI
import uuid
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.evaluate.bert_score import get_best_sentence
from src.evaluate.FLORES101.scoring import calcuate_bleu_score
from src.orchestration.state_manager import StateManager


class BaseMethod(ABC):
    """Base class for all methods."""
    def __init__(self, config):
        """Initialize base method with common attributes."""
        self.config = config
        self.config.validate()
        
        # Common configuration
        self.tasks = config.tasks
        self.test_tasks = config.test_tasks
        self.task_weights = config.task_weights
        self.seed = config.seed
        self.combine_method = config.combine_method
        self.model_name_or_path = config.model_name_or_path
        self.llm_base_url = config.llm_base_url
        self.pools = config.pools
        self.max_workers = len(self.llm_base_url)
        self.plot_enabled = config.plot_enabled
        
        # Global state tracking
        self.global_max_fitness_score = -100
        self.global_min_fitness_score = 100
        self.global_max_fitness_path = ""
        self.global_min_fitness_path = ""
        self.global_max_task_scores = {}
        self.global_min_task_scores = {}
        self.global_max_fitness_weight = dict()
        self.global_min_fitness_weight = dict()
        
        # Early stopping state
        self.patience_flag = True
        self.global_patience_counter = 0
        self.early_stop = config.early_stop
        self.early_stop_iter = config.early_stop_iter
        
        # Initialize workspace and state manager
        self.init_workspace()
        self.state_manager = StateManager(self.workspace)
        self.init_models()
        
        # Save initial config
        self.config.save(self.workspace)

    @property
    def state(self) -> Dict:
        return self.state_manager.state

    @state.setter
    def state(self, value: Dict) -> None:
        self.state_manager.state = value
    
    def init_workspace(self) -> None:
        """Initialize the workspace with a standardized directory structure."""
        model = self.model_name_or_path.split("/")[-1]
        self.id = uuid.uuid4().hex
        task_name = "_".join(self.tasks)
        method_name = self.__class__.__name__.lower()
        
        self.workspace = os.path.join(
            f"{method_name}_workspace",
            task_name,
            f"N{self.config.N}_{self.combine_method.value}",
            model,
            f"{method_name.upper()}-{self.id}"
        )
        logger.info(f"Workspace: {self.workspace}")
        os.makedirs(self.workspace, exist_ok=True)
    
    def init_models(self) -> None:
        """Initialize OpenAI API clients for each base URL."""
        self.llms = [
            OpenAI(
                base_url=base_url,
                api_key=f"{self.__class__.__name__.lower()}_api_key"
            ) for base_url in self.llm_base_url
        ]
    
    def update_global(self, id: str, fitness_score: float, path: str, task_scores: Dict[str, float]) -> None:
        """Update global state with new fitness information."""
        logger.info(f"Individual {id} fitness: {fitness_score:.4f}")
        if fitness_score > self.global_max_fitness_score:
            self.global_max_fitness_score = fitness_score
            self.global_max_fitness_path = path
            self.global_max_task_scores = task_scores.copy()
            self.global_max_fitness_weight = load_lora_weight(path)
            logger.info(f"Global max updated: {self.global_max_fitness_score:.4f}")
            if task_scores:
                logger.info("Best individual task scores:")
                for task, score in task_scores.items():
                    logger.info(f"  - Task {task}: {score:.4f}")
            self.patience_flag = False
            
        if fitness_score < self.global_min_fitness_score:
            self.global_min_fitness_score = fitness_score
            self.global_min_fitness_path = path
            self.global_min_task_scores = task_scores.copy()
            self.global_min_fitness_weight = load_lora_weight(path)
            logger.info(f"Global min updated: {self.global_min_fitness_score:.4f}")

    def get_global_state_snapshot(self) -> Dict[str, Any]:
        return {
            "max_score": self.global_max_fitness_score,
            "max_path": self.global_max_fitness_path,
            "min_score": self.global_min_fitness_score,
            "min_path": self.global_min_fitness_path,
        }

    def save_final_state(self, individuals: List, time: float) -> None:
        """Save final state and generate plots."""
        # Calculate weighted test scores
        weighted_scores = {
            individual.id: sum(
                self.state['test'][task][individual.id] * self.task_weights[idx]
                for idx, task in enumerate(self.test_tasks)
            )
            for individual in individuals
        }
        
        test_id = max(weighted_scores, key=weighted_scores.get)
        test_score = weighted_scores[test_id]

        self.state['final'] = {
            "test_id": test_id,
            "test_score": test_score,
            "total_time": time,
        }
        
        # Save state
        self.state_manager.save()
        logger.info(
            f"Best individual id: {test_id}, "
            f"Test performance: {test_score:.4f}"
        )
        
    def evaluate_single_task_ppl(self, individuals: List, task: str, split: str = "valid") -> Dict[str, Any]:
        perplexity = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = dict()
            for idx, individual in enumerate(individuals):
                future = executor.submit(
                    individual.fitness,
                    task=task,
                    llm=self.llms[idx % len(self.llms)],
                    lora_path=individual.weight_path,
                    split=split,
                    calculate_ppl=True,
                    return_predictions=False,
                )
                futures[future] = idx
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    perplexity[idx] = {
                        "id": result["id"],
                        "score": result['score'],
                        "perplexity": result['perplexity'],
                        "path": result['path']
                    }
                    logger.info(f"ID: {result['id']}, Perplexity: {result['perplexity']:.4f}")
                except Exception as e:
                    logger.error(f"Error processing future result: {str(e)}")
        
        return perplexity
    
    def evaluate_single_task(self, individuals: List, task: str, split: str = "valid") -> Dict[str, Any]:
        """Evaluate individuals on a single task."""
        task_scores = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for idx, individual in enumerate(individuals):
                futures.append(
                    executor.submit(
                        individual.fitness,
                        task=task,
                        llm=self.llms[idx % len(self.llms)],
                        lora_path=individual.weight_path,
                        split=split
                    )
                )
            #TODO: try to fix parallel bug, delete as_completed
            for future in as_completed(futures):
                try:
                    result = future.result()
                    task_scores[result["id"]] = {
                        "score": result["score"],
                        "path": result["path"]
                    }
                except Exception as e:
                    logger.error(f"Error processing future result: {str(e)}")
                
        return task_scores
    
    def compute_weighted_score(self, task_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict]:
        """Compute the weighted score for the individuals in all tasks."""
        weighted_scores = dict()
        
        for individual_id in task_scores[self.tasks[0]].keys():
            scores = {}
            total_score = 0
            assert hasattr(self, "task_weights"), logger.error("Task weights are not set.")
            assert hasattr(self, "tasks"), logger.error("Tasks are not set.")
            for task, weight in zip(self.tasks, self.task_weights):
                task_score = task_scores[task][individual_id]['score']
                scores[task] = task_score
                total_score += weight * task_score
                
            weighted_scores[individual_id] = {
                "weighted_score": total_score,
                "task_scores": scores,
                "path": task_scores[self.tasks[0]][individual_id]['path']
            }
            
        return weighted_scores
    
    def evaluate(self, individuals: List, split: str = "valid") -> Dict[str, Any]:
        """Evaluate the models."""
        logger.info(f"Start evaluating {len(individuals)} individuals on {len(self.tasks)} tasks...")
        
        if split != "valid":
            logger.warning(f"Evaluate split is not valid, got {split}.")
        
        # 1. Evaluate on each task separately
        all_task_scores = dict()
        for task in self.tasks:
            logger.info(f"Evaluating on task: {task}")
            all_task_scores[task] = self.evaluate_single_task(individuals=individuals, task=task, split=split)
        
        # 2. Compute the weighted score
        weighted_scores = self.compute_weighted_score(all_task_scores)
        
        # 3. Update individual fitness scores
        for individual in individuals:
            if individual.id in weighted_scores:
                individual.update_fitness(tasks=self.tasks, task_weights=self.task_weights)
                     
        # 4. Log results
        for individual_id, result in weighted_scores.items():
            # log detailed information
            logger.info(f"Individual {individual_id}:")
            logger.info(f"  Weighted score: {result['weighted_score']:.4f}")
            for task, score in result['task_scores'].items():
                logger.info(f"  - Task {task}: {score:.4f}")
                
        return weighted_scores
    
    def test(self, individuals: List, split: str = "test") -> None:
        """Test the models."""
        if 'test' not in self.state:
            self.state['test'] = dict()
            
        if split != "test":
            logger.warning(f"Test split is not valid, got {split}.")
            
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for idx, individual in enumerate(individuals):
                    individual.evaluated = False
                    futures.append(
                        executor.submit(
                            individual.fitness,
                            task=self.task,
                            llm=self.llms[idx % len(self.llms)],
                            lora_path=individual.weight_path,
                            split=split
                        )
                    )
                    
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        self.state['test'][result["id"]] = result["score"]
                        logger.info(f"Test ID: {result['id']}, Test Score: {result['score']:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error processing future result: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error in test method: {str(e)}")
    
    def ensemble_test(self, individuals: List, split: str = "test") -> Dict[str, Any]:
        """Ensemble test the models.
        This method:
        1. Collects predictions from all individuals for each test sample.
        2. Use majority voting to determine the final answer
        3. Computes and stores the ensemble accuracy
        
        Args:
            individuals (List): The individuals to be tested.
            split (str): The split to be tested. (default: "test")
        Returns:
            Dict[str, Any]: The ensemble test results.
        """
        if self.test_tasks != self.tasks:
            self.test_tasks = list(set(self.tasks + self.test_tasks))
            self.task_weights = [1.0 / len(self.test_tasks) for _ in range(len(self.test_tasks))]
            logger.warning(f"Test tasks and tasks are not consistent, merging them into {self.test_tasks}.")
            
        if 'ensemble_test' not in self.state:
            self.state['ensemble_test'] = dict()

        if 'test' not in self.state:
            self.state['test'] = dict()
            for task in self.test_tasks:
                self.state['test'][task] = dict()
            
        if split != "test":
            logger.warning(f"Ensemble test split is not valid, got {split}.")
        
        # 1. Evaluate on each task separately
        task_predictions = {task: [] for task in self.test_tasks}
        task_scores = {task: dict() for task in self.test_tasks}
        
        try:
            for task in self.test_tasks:
                logger.info(f"Ensemble test on task: {task}")
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for idx, individual in enumerate(individuals):
                        individual.evaluated[task] = False
                        futures.append(
                            executor.submit(
                                individual.fitness,
                                task=task,
                                llm=self.llms[idx%len(self.llms)],
                                lora_path=individual.weight_path,
                                split=split,
                                return_predictions=True,
                            )
                        )
                
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            task_scores[task][result["id"]] = result["score"]
                            self.state['test'][task][result['id']] = result['score']
                            task_predictions[task].append(dict(
                                id=result["id"],
                                score=result["score"],
                                predictions=result["predictions"]
                            ))
                            logger.info(f"Test {task} - ID: {result['id']}, Score: {result['score']:.4f}")
                            
                        except Exception as e:
                            logger.error(f"Error processing future result: {str(e)}")
            
            ensemble_results = {
                "task_results": {},
                "weighted_results": {}
            }
            
            sorted_individuals = {
                task: sorted(individuals, key=lambda x: x.task_scores[task], reverse=True)
                for task in self.test_tasks
            }
            
            for task in self.test_tasks:
                ensemble_results["task_results"][task] = {}
                for k in range(2, len(sorted_individuals[task])+1):
                    top_k_individuals = sorted_individuals[task][:k]
                    top_k_preds = [individual.predictions[task] for individual in top_k_individuals]
                    # different task has different scoring method
                    logger.info(f"Task {task} - Top-{k} Ensemble Test Start")
                    result = self._majority_vote(task=task, results=top_k_preds)
                    ensemble_results["task_results"][task][f"top_{k}"] = {
                        'score': result['score'],
                        'num_models': k,
                        'ensemble_ids': [individual.id for individual in top_k_individuals]
                    }
                    logger.info(f"Task {task} - Top-{k} Ensemble Test Score: {result['score']:.4f}")
            
            # 3. Calculate the weighted results from top 2 to top N
            for k in range(2, len(individuals)+1):
                weighted_score = 0
                task_scores = dict()
                if len(self.test_tasks) != len(self.task_weights):
                    logger.warning(f"Test tasks and task weights are not consistent, got {len(self.test_tasks)} and {len(self.task_weights)}.")
                    self.task_weights = [1.0 / len(self.test_tasks) for _ in range(len(self.test_tasks))]
                    
                for task, weight in zip(self.test_tasks, self.task_weights):
                    task_score = ensemble_results["task_results"][task][f"top_{k}"]['score']
                    task_scores[task] = task_score
                    weighted_score += weight * task_score
                
                ensemble_results["weighted_results"][f"top_{k}"] = {
                    "score": weighted_score,
                    "task_scores": task_scores,
                    "num_models": k,
                    "ensemble_ids": ensemble_results["task_results"][task][f"top_{k}"]['ensemble_ids']
                }
                logger.info(f"Top-{k} WeightedEnsemble Test Score: {weighted_score:.4f}")
                    
            self.state['ensemble_test'] = ensemble_results
            
        except Exception as e:
            logger.error(f"Error in ensemble test method: {str(e)}")
        
        self.state_manager.save()
        
        return ensemble_results
    
    def generate_pair_sequences(self, pools: List[str], n_samples: int, seed: Optional[int] = None) -> List:
        if seed is None:
            seed = self.config.seed
        
        all_pairs = []
        pool_size = len(pools)
        for i in range(pool_size):
            for j in range(i+1, pool_size):
                all_pairs.append((pools[i], pools[j]))
        
        r = random.Random(seed)
        r.shuffle(all_pairs)
        
        if n_samples > len(all_pairs):
            complete_sets = n_samples // len(all_pairs)
            remainder = n_samples % len(all_pairs)
            
            expert_pairs = all_pairs * complete_sets + all_pairs[:remainder]
        else:
            expert_pairs = all_pairs[:n_samples]
        
        self.state["init_expert_pairs"] = []
        for expert_pair in expert_pairs:
            expert_1, expert_2 = expert_pair[0].split("/")[-1], expert_pair[1].split("/")[-1]
            logger.info(f"Init -> Expert pair: {expert_1} and {expert_2}")
            self.state["init_expert_pairs"].append([expert_1, expert_2])
            
        return expert_pairs
    
    def merge_lora_weights(
        self, lora_state_dicts: List[dict], weights: Optional[List[float]] = None, 
        method: Literal[CombineMethod.TIES, CombineMethod.LINEAR, CombineMethod.DARE_TIES, CombineMethod.DARE_LINEAR, CombineMethod.BLXALPHA, CombineMethod.RANDOM] = CombineMethod.TIES,
        density: float = 0.7, alpha: float = 0.5,
        majority_sign_method: str = "total"
    ) -> Dict:
        if isinstance(method, str):
            method = CombineMethod(method)
            
        logger.info(f"Merge {len(lora_state_dicts)} lora weights with method: {method}")
        if weights is None or len(weights) != len(lora_state_dicts):
            # generate random weights
            weights = [random.random() for _ in range(len(lora_state_dicts))]
        
        weights = torch.tensor(weights)
        weights = weights / weights.sum()

        if method == CombineMethod.TIES:
            merge_fn = process_ties
        elif method == CombineMethod.LINEAR:
            merge_fn = process_linear
        elif method == CombineMethod.DARE_TIES:
            merge_fn = process_dare_ties
        elif method == CombineMethod.DARE_LINEAR:
            merge_fn = process_dare_linear
        elif method == CombineMethod.BLXALPHA:
            merge_fn = process_blxalpha
        elif method == CombineMethod.RANDOM:
            merge_fn = process_random
        else:
            raise ValueError(f"Invalid method: {method}")
        
        all_keys = set()
        for lora_state_dict in lora_state_dicts:
            all_keys.update(lora_state_dict.keys())
            
        a_keys = [key for key in all_keys if key.startswith("a")]
        b_keys = [key for key in all_keys if key.startswith("b")]
        
        merged_state_dict = dict()
        
        for key in a_keys:
            tensors = []
            for lora_dict in lora_state_dicts:
                if key in lora_dict:
                    tensors.append(lora_dict[key])
            
            if len(tensors) > 0:
                if method in [CombineMethod.TIES, CombineMethod.DARE_TIES]:
                    merged_state_dict[key] = merge_fn(tensors, weights, density, majority_sign_method)
                elif method == CombineMethod.LINEAR:
                    merged_state_dict[key] = merge_fn(tensors, weights)
                elif method == CombineMethod.DARE_LINEAR:
                    merged_state_dict[key] = merge_fn(tensors, weights, density)
                elif method == CombineMethod.BLXALPHA:
                    merged_state_dict[key] = merge_fn(tensors, weights, alpha)
                elif method == CombineMethod.RANDOM:
                    merged_state_dict[key] = merge_fn(tensors, weights, weighted_by_magnitude=True, temperature=1.0)
                else:
                    raise ValueError(f"Invalid method: {method}")
            
        for key in b_keys:
            tensors = []
            for lora_dict in lora_state_dicts:
                if key in lora_dict:
                    tensors.append(lora_dict[key])
            
            if len(tensors) > 0:
                if method in [CombineMethod.TIES, CombineMethod.DARE_TIES]:
                    merged_state_dict[key] = merge_fn(tensors, weights, density, majority_sign_method)
                elif method == CombineMethod.LINEAR:
                    merged_state_dict[key] = merge_fn(tensors, weights)
                elif method == CombineMethod.DARE_LINEAR:
                    merged_state_dict[key] = merge_fn(tensors, weights, density)
                elif method == CombineMethod.BLXALPHA:
                    merged_state_dict[key] = merge_fn(tensors, weights, alpha)
                elif method == CombineMethod.RANDOM:
                    merged_state_dict[key] = merge_fn(tensors, weights, weighted_by_magnitude=True, temperature=1.0)
                else:
                    raise ValueError(f"Invalid method: {method}")
            
        return merged_state_dict
    
    def linear_combination(self, lora_1: Dict, lora_2: Dict, alpha: Optional[float] = None, seed: Optional[int] = None) -> Dict:
        """Compute a linear combination of two LoRA weight dictionaries.
        
        This method combines two LoRA weight dictionaries using a weighted sum:
        merged = alpha * lora_1 + (1-alpha) * lora_2
        
        If alpha is not provided, it is randomly generated based on a hash of the 
        LoRA weights and the provided seed to ensure reproducibility.
        
        Args:
            lora_1 (Dict): First LoRA weight dictionary containing model parameters
            lora_2 (Dict): Second LoRA weight dictionary containing model parameters
            alpha (Optional[float]): Weight coefficient for lora_1, between 0 and 1.
                                   If None, randomly generated.
            seed (Optional[int]): Random seed for reproducible alpha generation.
                                If None, uses self.seed or defaults to 42.
            
        Returns:
            Dict: New LoRA weight dictionary containing the linear combination
                 of the input weights.
        """
        assert set(lora_1.keys()) == set(lora_2.keys()), "The two lora configs should have the same keys."
        
        if seed is None:
            seed = self.seed if self.seed is not None else 42
        
        if alpha is None:
            sorted_keys = sorted(lora_1.keys())
            weight_sum = int(sum(abs(lora_1[key]).sum().item() + abs(lora_2[key]).sum().item() for key in sorted_keys) * 1e7)
            logger.info(f"The weight sum is {weight_sum}.")
            assert weight_sum > 0, f"The weight sum ({weight_sum}) should be greater than 0."
            base_seed = seed if seed is not None else 42
            combined_seed = hash(weight_sum) + base_seed
            rng = random.Random(combined_seed)
            alpha = rng.random()
        beta = 1 - alpha
        
        merged_state_dict = dict()
        for key in lora_1.keys():
            merged_state_dict[key] = alpha * lora_1[key] + beta * lora_2[key]
            
        return merged_state_dict
    
    def _majority_vote(self, task: str, results: List[Dict]) -> Dict:
        """Use majority voting to determine the final answer.
        
        When multiple answers have the same highest votes:
        - If reference answer is among them -> 0.5 point
        - If reference answer is not among them -> 0 point
        When one answer has the highest votes:
        - If it's the reference answer -> 1 point
        - If it's not the reference answer -> 0 point
        Args:
            results (List[Dict]): The results of the individuals.
            
        Returns:
            Dict: The final answer and the number of votes for each answer.
        """
        if task == "drop":
            return self._majority_vote_drop(task=task, results=results)
        elif task == "flores101" or task == "flores37":
            return self._majority_vote_similarity(task=task, results=results)
        
        vote_counts = defaultdict(lambda: {'predicted_answers': [], 'reference_answer': None})
            
        for result in results:
            for question_id, pred in result.items():
                vote_counts[question_id]['reference_answer'] = pred['reference_answer']
                vote_counts[question_id]['predicted_answers'].append(pred['predicted_answer'])
        
        total_count = 0
        correct_count = 0
        final_predictions = {}
        voting_process = []
        
        for question_id, votes in vote_counts.items():
            predictions = votes['predicted_answers']
            reference_answer = votes['reference_answer']
            vote_distribution = Counter(predictions)
            
            max_votes = max(vote_distribution.values())
            majority_answers = [
                answer for answer, count in vote_distribution.items() 
                if count == max_votes
            ]
            
            if len(majority_answers) == 1:
                is_correct = (majority_answers[0] == reference_answer)
                if is_correct:
                    correct_count += 1
                final_answer = majority_answers[0]
            elif len(majority_answers) > 1:
                is_correct = (reference_answer in majority_answers)
                if is_correct:
                    correct_count += 0.5
                final_answer = reference_answer if is_correct else majority_answers[0]
            else:
                is_correct = False
                final_answer = majority_answers[0]

            total_count += 1
            
            voting_process.append({
                'question_id': question_id,
                'vote_distribution': dict(vote_distribution),
                'majority_answers': majority_answers,
                'selected_answer': final_answer,
                'reference_answer': reference_answer,
                'is_correct': is_correct,
                "is_tie": len(majority_answers) > 1,
            })
            
            final_predictions[question_id] = {
                'reference_answer': reference_answer,
                'majority_answers': majority_answers,
                'selected_answer': final_answer,
                'is_correct': is_correct,
                "is_tie": len(majority_answers) > 1,
            }
        
        voting_record = {
            'total_questions': total_count,
            'correct_answers': correct_count,
            'accuracy': correct_count / total_count if total_count > 0 else 0,
            'voting_process': voting_process
        }
        voting_record_path = os.path.join(self.workspace, 'voting_records', f"{task}_top-{len(results)}.json")
        os.makedirs(os.path.dirname(voting_record_path), exist_ok=True)
        with open(voting_record_path, "w") as f:
            f.write(json.dumps(voting_record, indent=4, ensure_ascii=False))
            
        return {
            'score': correct_count / total_count if total_count > 0 else 0,
            'predictions': final_predictions
        }

    def _majority_vote_drop(self, task: str, results: List[Dict]) -> Dict:
        score_mapping = dict()
        for result in results:
            for question_id, pred in result.items():
                if question_id not in score_mapping:
                    score_mapping[question_id] = {}
                    score_mapping[question_id]['em_scores'] = []
                    score_mapping[question_id]['f1_scores'] = []
                score_mapping[question_id]['em_scores'].append(pred['score']['em_score'])
                score_mapping[question_id]['f1_scores'].append(pred['score']['f1_score'])    
        em_scores = []
        f1_scores = []
        voting_process = []
        final_predictions = {}
        for key, value in score_mapping.items():
            em_scores.append(
                max(value['em_scores'])
            )    
            f1_scores.append(
                max(value['f1_scores'])
            )
            voting_process.append({
                "question_id": key,
                "majority_em_score": max(value['em_scores']),
                "majority_f1_score": max(value['f1_scores']),
                "reference_answer": value.get('reference_answer', None),
                "predicted_answer": value.get('predicted_answers', None),
                "is_correct": None,
            })
            final_predictions[key] = {
                "majority_em_score": max(value['em_scores']),
                "majority_f1_score": max(value['f1_scores']),
                "reference_answer": value.get('reference_answer', None),
                "predicted_answer": value.get('predicted_answers', None),
                "is_correct": None,
            }

        ave_em_score = sum(em_scores) / len(em_scores)
        ave_f1_score = sum(f1_scores) / len(f1_scores)
        # calculate score
        voting_record = {
            "total_questions": len(score_mapping),
            "process": voting_process,
        }
        voting_record_path = os.path.join(self.workspace, 'voting_records', f"{task}_top-{len(results)}.json")
        os.makedirs(os.path.dirname(voting_record_path), exist_ok=True)
        with open(voting_record_path, "w") as f:
            f.write(json.dumps(voting_record, indent=4, ensure_ascii=False))

        return {
            "score": ave_em_score,
            "predictions": final_predictions,
        }
    
    def _majority_vote_similarity(self, task: str, results: List[Dict]) -> Dict:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        def single_vote(question_id: str, votes: Dict) -> Dict:
            reference_answer = votes['reference_answer']
            bert_score_result = get_best_sentence(sentence_list=votes['predicted_answers'])
            best_sentence, best_score = bert_score_result['best_sentence']
            bleu_score = calcuate_bleu_score(
                tokenizer=tokenizer,
                predicted=best_sentence,
                reference=reference_answer,
            )
            return {
                "question_id": question_id,
                "majority_answers": best_sentence,
                "majority_bert_score": best_score,
                "majority_bleu_score": bleu_score,
                "reference_answer": reference_answer,
                "predicted_answers": votes['predicted_answers'],
            }
        
        vote_records = defaultdict(lambda: {'predicted_answers': [], 'reference_answer': None})
        
        for result in results:
            for question_id, pred in result.items():
                vote_records[question_id]['reference_answer'] = pred['reference_answer']
                vote_records[question_id]['predicted_answers'].append(pred['predicted_answer'])
                
        final_predictions = {}
        voting_process = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    single_vote, question_id, votes
                ) for question_id, votes in vote_records.items()
            ]
            for future in as_completed(futures):
                single_vote_result = future.result()
                voting_process.append({
                    **single_vote_result,
                    "is_correct": None,
                })
                final_predictions[question_id] = single_vote_result
        
        ave_bleu_score = sum(single_vote_result['majority_bleu_score'] for single_vote_result in voting_process) / len(voting_process)
        voting_record = {
            "total_questions": len(vote_records),
            "process": voting_process,
        }
        voting_record_path = os.path.join(self.workspace, 'voting_records', f"{task}_top-{len(results)}.json")
        os.makedirs(os.path.dirname(voting_record_path), exist_ok=True)
        with open(voting_record_path, "w") as f:
            f.write(json.dumps(voting_record, indent=4, ensure_ascii=False))
        
        return {
            "score": ave_bleu_score,
            "predictions": final_predictions,
        }
            
    def get_top_k_individuals(self, individuals: List, k: int) -> List:
        """Get top-k individuals with highest fitness scores.
        
        Args:
            individuals (List): List of individuals to select from
            k (int): Number of top individuals to return
            
        Returns:
            List: Top-k individuals sorted by fitness score in descending order
        """
        # Sort individuals by fitness score in descending order
        sorted_individuals = sorted(individuals, key=lambda x: x.fitness_score, reverse=True)
        
        # Return top k individuals (or all if k > len)
        return sorted_individuals[:min(k, len(sorted_individuals))]
    
    @abstractmethod
    def search(self) -> None:
        """Execute the optimization search process."""
        pass
    
    def plot_optimization_curves(self):
        """Plot optimization curves including:
        - Global max fitness score
        - Average fitness score
        - Per-task scores
        """
        steps = sorted([int(k.split('_')[1]) for k in self.state.keys() if k.startswith('step_')])
        
        # Prepare data
        global_max_scores = []
        average_scores = []
        task_scores = {task: [] for task in self.tasks}
        
        for step in steps:
            step_data = self.state[f'step_{step}']
            global_max_scores.append(step_data['global_max_fitness_score'])
            average_scores.append(step_data['average_fitness_score'])
            
            if 'task_scores' in step_data:
                for task in self.tasks:
                    task_scores[task].append(step_data['task_scores'][task]['avg'])
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot global max and average scores
        plt.plot(steps, global_max_scores, 'b-', label='Global Max', marker='o')
        plt.plot(steps, average_scores, 'r--', label='Average', marker='s')
        
        # Plot task scores if available
        colors = ['g', 'm', 'c', 'y']
        for i, task in enumerate(self.tasks):
            if task_scores[task]:
                plt.plot(steps, task_scores[task], f'{colors[i%len(colors)]}--', 
                        label=f'Task: {task}', marker='.')
        
        plt.xlabel('Step')
        plt.ylabel('Score')
        plt.title('Optimization Progress')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_dir = os.path.join(self.workspace, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'optimization_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Optimization curves saved to {plot_dir}/optimization_curves.png")

    def plot_score_distributions(self):
        """Plot score distributions including:
        - Box plots for each task
        - Score histograms for each task
        """
        if 'test' not in self.state:
            logger.warning("No test data available for plotting score distributions")
            return
            
        # Prepare data
        task_scores = {}
        for task in self.test_tasks:
            if task in self.state['test']:
                task_scores[task] = list(self.state['test'][task].values())
        
        if not task_scores:
            logger.warning("No data available for plotting score distributions")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Box plot
        ax1.boxplot([task_scores[task] for task in self.test_tasks], labels=self.test_tasks)
        ax1.set_title('Score Distribution by Task')
        ax1.set_ylabel('Score')
        ax1.grid(True)
        
        # Histograms
        colors = ['b', 'r', 'g', 'm']
        for i, task in enumerate(self.test_tasks):
            ax2.hist(task_scores[task], alpha=0.5, label=task, 
                    color=colors[i%len(colors)], bins=20)
        ax2.set_title('Score Distribution Density by Task')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = os.path.join(self.workspace, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'score_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Score distributions saved to {plot_dir}/score_distributions.png")

    def plot_correlation_matrix(self):
        """Plot correlation matrix analyzing:
        - Correlation between tasks
        - Correlation between tasks and total score
        """
        if 'test' not in self.state:
            logger.warning("No test data available for plotting correlation matrix")
            return
            
        # Prepare data
        data = {}
        for task in self.test_tasks:
            if task in self.state['test']:
                data[task] = []
                for individual_id in self.state['test'][task].keys():
                    data[task].append(self.state['test'][task][individual_id])
        
        if not data:
            logger.warning("No data available for plotting correlation matrix")
            return
            
        # Convert to DataFrame and calculate correlation
        df = pd.DataFrame(data)
        df['Total'] = sum(df[task] * weight for task, weight in zip(self.test_tasks, self.task_weights))
        corr = df.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot correlation matrix
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add colorbar
        plt.colorbar(im)
        
        # Add labels
        labels = list(corr.columns)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add correlation values
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        plt.title('Correlation Matrix of Task Scores')
        
        # Save plot
        plot_dir = os.path.join(self.workspace, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Correlation matrix saved to {plot_dir}/correlation_matrix.png")

    def plot_ensemble_performance(self):
        """Plot ensemble performance analysis:
        - Performance vs number of models in ensemble for each task
        - Shows how performance changes with ensemble size
        """
        if 'ensemble_test' not in self.state or 'weighted_results' not in self.state['ensemble_test']:
            logger.warning("No ensemble test data available for plotting")
            return
            
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Get data from weighted_results
        weighted_results = self.state['ensemble_test']['weighted_results']
        
        # Prepare data
        sizes = []
        overall_scores = []
        task_scores = {task: [] for task in self.test_tasks}
        
        # Extract data for each ensemble size
        for ensemble_info in weighted_results.values():
            sizes.append(ensemble_info['num_models'])
            overall_scores.append(ensemble_info['score'])
            for task in self.test_tasks:
                task_scores[task].append(ensemble_info['task_scores'][task])
        
        # Sort all data by ensemble size
        sorted_indices = sorted(range(len(sizes)), key=lambda k: sizes[k])
        sizes = [sizes[i] for i in sorted_indices]
        overall_scores = [overall_scores[i] for i in sorted_indices]
        for task in self.test_tasks:
            task_scores[task] = [task_scores[task][i] for i in sorted_indices]
        
        # Plot task-wise performance
        colors = ['r', 'g', 'b', 'm', 'c', 'y']
        for i, task in enumerate(self.test_tasks):
            ax1.plot(sizes, task_scores[task],
                    color=colors[i % len(colors)],
                    marker='o',
                    label=task)
        
        ax1.set_xlabel('Number of Models in Ensemble')
        ax1.set_ylabel('Score')
        ax1.set_title('Task-wise Performance vs Ensemble Size')
        ax1.legend()
        ax1.grid(True)
        
        # Plot overall weighted performance
        ax2.plot(sizes, overall_scores, 'b-', marker='o')
        ax2.set_xlabel('Number of Models in Ensemble')
        ax2.set_ylabel('Weighted Score')
        ax2.set_title('Overall Weighted Performance vs Ensemble Size')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = os.path.join(self.workspace, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'ensemble_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log some statistics
        logger.info(f"Ensemble performance plots saved to {plot_dir}/ensemble_performance.png")
        logger.info("Performance summary:")
        logger.info(f"Best overall score: {max(overall_scores):.4f} (with {sizes[overall_scores.index(max(overall_scores))]} models)")
        for task in self.test_tasks:
            best_score = max(task_scores[task])
            best_size = sizes[task_scores[task].index(best_score)]
            logger.info(f"Best {task} score: {best_score:.4f} (with {best_size} models)")

    def generate_plots(self):
        """Generate all plots."""
        logger.info("Generating plots...")
        
        try:
            self.plot_optimization_curves()
        except Exception as e:
            logger.error(f"Error generating optimization curves: {e}")
            
        try:
            self.plot_score_distributions()
        except Exception as e:
            logger.error(f"Error generating score distributions: {e}")
            
        try:
            self.plot_correlation_matrix()
        except Exception as e:
            logger.error(f"Error generating correlation matrix: {e}")
            
        try:
            self.plot_ensemble_performance()
        except Exception as e:
            logger.error(f"Error generating ensemble performance plots: {e}")
        
        logger.info("All plots generated successfully.")