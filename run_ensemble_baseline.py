import argparse

from src.ensemble_baseline import EnsembleBaselineConfig, EnsembleBaselineMethod
from src.utils import get_base_url, get_lora_pools


def parse_args():
    parser = argparse.ArgumentParser(description="Run ensemble baseline on LoRA pools (no search).")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--tasks", type=str, nargs="+", required=True)
    parser.add_argument("--test_tasks", type=str, nargs="+", required=True)
    parser.add_argument("--task_weights", type=float, nargs="+", required=True)
    parser.add_argument("--combine_method", type=str, default="ties")
    parser.add_argument("--ports", type=int, nargs="+", default=[18177])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_enabled", action="store_true")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_iter", type=int, default=5)
    parser.add_argument(
        "--pairwise_merge_before_ensemble",
        action="store_true",
        help="If set, first merge all unique pool pairs and ensemble on merged adapters only.",
    )
    parser.add_argument(
        "--workspace_prefix",
        type=str,
        default=None,
        help="Optional base directory prefix for workspace outputs",
    )
    parser.add_argument(
        "--max_valid_samples", type=int, default=200,
        help="Maximum number of validation samples per task",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total = sum(args.task_weights)
    task_weights = [w / total for w in args.task_weights]

    config = EnsembleBaselineConfig(
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
        pairwise_merge_before_ensemble=args.pairwise_merge_before_ensemble,
        workspace_prefix=args.workspace_prefix,
        max_valid_samples=args.max_valid_samples,
    )

    method = EnsembleBaselineMethod(config)
    method.search()


if __name__ == "__main__":
    main()
