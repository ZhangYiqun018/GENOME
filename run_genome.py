import argparse
import os
import time
from typing import List

from src.genome import Genome, GenomeConfig, Individual
from src.utils import get_base_url, get_lora_pools


def parse_args():
    parser = argparse.ArgumentParser(description="Genome Search with vLLM")

    parser.add_argument(
        "--model_path", type=str, required=True,
    )
    parser.add_argument(
        "--lora_dir", type=str, required=True,
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", required=True,
        help="List of tasks to evaluate (e.g., mmlu gsm8k arc_c)"
    )
    parser.add_argument(
        "--test_tasks", type=str, nargs="+", required=True,
        help="List of tasks to evaluate (e.g., mmlu gsm8k arc_c)"
    )
    parser.add_argument(
        "--task_weights", type=float, nargs="+",
        help="Weights for each task (must match number of tasks)"
    )
    parser.add_argument(
        "--combine_method", type=str, required=True, default="ties",
    )
    parser.add_argument(
        "--cross_method", type=str, required=True, default="ties",
    )
    parser.add_argument('--plot_enabled', action='store_true', help='Enable plotting')
    # port 
    parser.add_argument(
        '--ports', type=int, nargs='+', default=[18177, 36048, 22246, 13732, 43782, 35293, 55779, 11435],
    )
    parser.add_argument(
        "--workspace_prefix", type=str, default=None,
        help="Optional base directory prefix for workspace outputs"
    )
    parser.add_argument(
        '--population_size', type=int, default=10, 
    )
    parser.add_argument(
        '--iters', type=int, default=50,
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--early_stop", action="store_true",
    )
    parser.add_argument(
        "--early_stop_iter", type=int, default=5,
    )

    # hyperparameters
    parser.add_argument("--cross_rate", type=float, default=0.8)
    parser.add_argument("--individual_mutation_rate", type=float, default=0.15)
    parser.add_argument("--gene_mutation_rate", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--elite_percent", type=float, default=0.02)

    args = parser.parse_args()
    # Normalize task weights
    total = sum(args.task_weights)
    args.task_weights = [w/total for w in args.task_weights]
    return args


def main():
    args = parse_args()

    config = GenomeConfig(
        tasks = args.tasks,
        test_tasks = args.test_tasks,
        task_weights = args.task_weights,
        model_name_or_path = args.model_path,
        N = args.population_size,
        max_iter = args.iters,
        llm_base_url = get_base_url(args.ports),
        pools = get_lora_pools(args.lora_dir),
        combine_method = args.combine_method,
        # hyper parameters
        cross_rate = args.cross_rate,
        cross_method = args.cross_method,
        plot_enabled = args.plot_enabled,
        individual_mutation_rate = args.individual_mutation_rate,
        gene_mutation_rate=args.gene_mutation_rate,
        sigma=args.sigma,
        elite_percent=args.elite_percent,
        early_stop=args.early_stop,
        early_stop_iter=args.early_stop_iter,
        seed=args.seed,
        method = "roulette",
        workspace_prefix=args.workspace_prefix,
    )
    genome = Genome(config=config)
    
    genome.search()

if __name__ == '__main__':
    main()
