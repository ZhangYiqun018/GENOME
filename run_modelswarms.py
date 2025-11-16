import os
import time
import argparse
from typing import List

from src.deploy_vllm import main as deploy_vllm
from src.swarms.config import PSOConfig
from src.swarms.particle import Particle
from src.swarms.pso import PSO
from src.utils import get_base_url, get_lora_pools

def parse_args():
    parser = argparse.ArgumentParser(description='Model Swarms (PSO) Search with vLLM')
    
    # required parameters
    parser.add_argument('--tasks', type=str, nargs="+", required=True, help='List of tasks to evaluate (e.g., mmlu gsm8k arc_c)')
    parser.add_argument('--test_tasks', type=str, nargs="+", required=True, help='List of tasks to evaluate (e.g., mmlu gsm8k arc_c)')
    parser.add_argument('--task_weights', type=float, nargs='+', help='Weights for each task (must match number of tasks)')
    parser.add_argument('--model_path', type=str, required=True, default="meta-llama/Meta-Llama-3-8B-Instruct", help='Path to the model')
    parser.add_argument('--lora_dir', type=str, required=True, default="lora_adapters", help='Directory containing LoRA adapters')
    parser.add_argument('--combine_method', type=str, required=True, default="ties", help='Combine method')
    parser.add_argument('--plot_enabled', action='store_true', help='Enable plotting')
    # ports
    parser.add_argument('--ports', type=int, nargs='+',
                        default=[18177, 36048, 22246, 13732, 43782, 35293, 55779, 11435],
                        help='List of ports for API servers')
    
    # PSO parameters
    parser.add_argument('--population_size', type=int, default=30,
                        help='Population size (N)')
    parser.add_argument('--max_iter', type=int, default=50,
                        help='Maximum iterations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--lambda_step', type=float, default=0.5,
                        help='Lambda step')
    parser.add_argument('--phi_lambda', type=float, default=0.95,
                        help='Phi lambda')
    parser.add_argument('--phi_inertia', type=float, default=0.2,
                        help='Phi inertia')
    parser.add_argument('--phi_cognitive', type=float, default=0.2,
                        help='Phi cognitive')
    parser.add_argument('--phi_social', type=float, default=0.2,
                        help='Phi social')
    parser.add_argument('--phi_repel', type=float, default=0.1,
                        help='Phi repel')
    
    # early stopping
    parser.add_argument('--early_stop', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--early_stop_iter', type=int, default=50,
                        help='Early stopping iteration threshold')
    parser.add_argument('--unable_random', action='store_true',
                        help='Disable random initialization')
    parser.add_argument(
        "--workspace_prefix", type=str, default=None,
        help="Optional base directory prefix for workspace outputs"
    )
    
    args = parser.parse_args()
    ## task_weights normalization
    total = sum(args.task_weights)
    args.task_weights = [w/total for w in args.task_weights]
    return args

def main():
    args = parse_args()
    
    # set number of workers
    max_workers = len(args.ports)
    
    # create PSO configuration
    config = PSOConfig(
        tasks=args.tasks,
        test_tasks=args.test_tasks,
        task_weights=args.task_weights,
        model_name_or_path=args.model_path,
        llm_base_url=get_base_url(args.ports),
        plot_enabled=args.plot_enabled,
        max_workers=max_workers,
        pools=get_lora_pools(args.lora_dir),
        N=args.population_size,
        max_iter=args.max_iter,
        early_stop=args.early_stop,
        early_stop_iter=args.early_stop_iter,
        unable_random=args.unable_random,
        seed=args.seed,
        combine_method=args.combine_method,
        lambda_step=args.lambda_step,
        phi_lambda=args.phi_lambda,
        phi_inertia=args.phi_inertia,
        phi_cognitive=args.phi_cognitive,
        phi_social=args.phi_social,
        phi_repel=args.phi_repel,
        workspace_prefix=args.workspace_prefix,
    )
    
    # initialize and run PSO
    pso = PSO(config)
    pso.search()

if __name__ == '__main__':
    main()
