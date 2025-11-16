import argparse

from src.merge3 import Merge3Config, Merge3LoRAMethod
from src.utils import get_base_url, get_lora_pools


def parse_args():
    parser = argparse.ArgumentParser(description="Run Merge3-style search on LoRA pools")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--test_tasks", type=str, nargs="+", required=True)
    parser.add_argument("--task_weights", type=float, nargs="+", required=True)
    parser.add_argument("--combine_method", type=str, default="ties")
    parser.add_argument("--ports", type=int, nargs="+", default=[18177])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsga_generations", type=int, default=5)
    parser.add_argument("--nsga_population", type=int, default=4)
    parser.add_argument("--genotype_dimension", type=int, default=2)
    parser.add_argument("--parent_sample_size", type=int, default=2)
    parser.add_argument("--variable_bounds", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--plot_enabled", action="store_true")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_iter", type=int, default=5)
    parser.add_argument(
        "--workspace_prefix", type=str, default=None,
        help="Optional base directory prefix for workspace outputs"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total = sum(args.task_weights)
    task_weights = [w / total for w in args.task_weights]

    config = Merge3Config(
        tasks=args.tasks,
        test_tasks=args.test_tasks,
        task_weights=task_weights,
        model_name_or_path=args.model_path,
        llm_base_url=get_base_url(args.ports),
        pools=get_lora_pools(args.lora_dir),
        combine_method=args.combine_method,
        plot_enabled=args.plot_enabled,
        early_stop=args.early_stop,
        early_stop_iter=args.early_stop_iter,
        seed=args.seed,
        nsga_generations=args.nsga_generations,
        nsga_population=args.nsga_population,
        genotype_dimension=args.genotype_dimension,
        parent_sample_size=args.parent_sample_size,
        variable_bounds=tuple(args.variable_bounds),
        save_intermediate=True,
        workspace_prefix=args.workspace_prefix,
    )

    method = Merge3LoRAMethod(config)
    method.search()


if __name__ == "__main__":
    main()
