from src.lorahub import LoraHub, LoraHubConfig

import argparse
import os
from typing import List
from src.utils import get_base_url, get_lora_pools


def parse_args():
    parser = argparse.ArgumentParser(description="LoraHub Search")
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--do_search", action="store_true", help="Enable search")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--test_tasks", type=str, nargs="+", required=True)
    parser.add_argument("--task_weights", type=float, nargs="+", help="Weights for each task (must match number of tasks)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_iter", type=int, default=5)
    parser.add_argument("--plot_enabled", action="store_true", help="Enable plotting")
    parser.add_argument("--combine_method", type=str, default="linear")
    parser.add_argument(
        '--ports', type=int, nargs='+', default=[9112],
    )
    parser.add_argument(
        "--workspace_prefix", type=str, default=None,
        help="Optional base directory prefix for workspace outputs"
    )
    parser.add_argument(
        "--max_valid_samples", type=int, default=200,
        help="Maximum number of validation samples per task",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = LoraHubConfig(
        N=args.N,
        max_iter=args.max_iter,
        model_name_or_path=args.model_path,
        do_search=args.do_search,
        pools=get_lora_pools(args.lora_dir),
        llm_base_url=get_base_url(args.ports),
        tasks=args.tasks,
        test_tasks=args.test_tasks,
        task_weights=args.task_weights,
        seed=args.seed,
        early_stop=args.early_stop,
        early_stop_iter=args.early_stop_iter,
        plot_enabled=args.plot_enabled,
        combine_method=args.combine_method,
        workspace_prefix=args.workspace_prefix,
        max_valid_samples=args.max_valid_samples,
    )
    
    lorahub = LoraHub(config=config)
    lorahub.search()
    

if __name__ == '__main__':
    main()
