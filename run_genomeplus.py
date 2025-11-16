from src.genomeplus import GenomePlus, GenomePlusConfig

import argparse
import os
from typing import List
from src.utils import get_base_url, get_lora_pools


def parse_args():
    parser = argparse.ArgumentParser(description="GenomePlus Search with vLLM")
    
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
        "--do_init", action="store_true", help="Whether to initialize"
    )
    parser.add_argument(
        "--task_weights", type=float, nargs="+",
        help="Weights for each task (must match number of tasks)"
    )
    parser.add_argument(
        "--combine_method", type=str, required=True, default="linear",
    )
    parser.add_argument(
        "--cross_method", type=str, required=True, default="ties",
    )
    parser.add_argument(
        "--plot_enabled", action="store_true", help="Enable plotting",
    )
    # port 
    parser.add_argument(
        '--ports', type=int, nargs='+', default=[18177, 36048, 22246, 13732, 43782, 35293, 55779, 11435],
    )
    parser.add_argument(
        "--workspace_prefix", type=str, default=None,
        help="Optional base directory prefix for workspace outputs"
    )
    parser.add_argument('--init_population_size', type=int, default=10)
    parser.add_argument('--max_population_size', type=int, default=20)

    parser.add_argument('--max_iter', type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_iter", type=int, default=5)
    parser.add_argument("--unable_random", action="store_true")

    parser.add_argument("--lambda_step", type=float, default=0.5)
    parser.add_argument("--phi_lambda", type=float, default=0.95)
    parser.add_argument("--phi_inertia", type=float, default=0.2)
    parser.add_argument("--phi_cognitive", type=float, default=0.2)
    parser.add_argument("--phi_social", type=float, default=0.2)
    parser.add_argument("--phi_repel", type=float, default=0.1)

    parser.add_argument("--cross_rate", type=float, default=0.8)
    parser.add_argument("--individual_mutation_rate", type=float, default=0.3)
    parser.add_argument("--gene_mutation_rate", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--elite_percent", type=float, default=0.1)
    parser.add_argument("--method", type=str, default="roulette")
    parser.add_argument("--selection_method", type=str, default="roulette")

    args = parser.parse_args()
    # Normalize task weights
    total = sum(args.task_weights)
    args.task_weights = [w/total for w in args.task_weights]
    return args

def main():
    args = parse_args()
    
    config = GenomePlusConfig(
        N_init=args.init_population_size,
        N_max=args.max_population_size,
        do_init=args.do_init,
        model_name_or_path=args.model_path,
        pools = get_lora_pools(args.lora_dir),
        llm_base_url=get_base_url(args.ports),
        tasks=args.tasks,
        test_tasks=args.test_tasks,
        task_weights=args.task_weights,
        max_iter=args.max_iter,
        combine_method=args.combine_method,
        cross_method=args.cross_method,
        plot_enabled=args.plot_enabled,
        lambda_step=args.lambda_step,
        phi_lambda=args.phi_lambda,
        phi_inertia=args.phi_inertia,
        phi_cognitive=args.phi_cognitive,
        phi_social=args.phi_social,
        phi_repel=args.phi_repel,
        unable_random=args.unable_random,
        cross_rate=args.cross_rate,
        individual_mutation_rate=args.individual_mutation_rate,
        gene_mutation_rate=args.gene_mutation_rate,
        sigma=args.sigma,
        elite_percent=args.elite_percent,
        seed=args.seed,
        early_stop=args.early_stop,
        early_stop_iter=args.early_stop_iter,
        method=args.method,
        selection_method=args.selection_method,
        workspace_prefix=args.workspace_prefix,
    )
    genomeplus = GenomePlus(config=config)
    genomeplus.search()
    

if __name__ == '__main__':
    main()
