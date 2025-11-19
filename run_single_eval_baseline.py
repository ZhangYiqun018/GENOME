import argparse

from src.single_eval_baseline import (
    SingleEvalBaselineConfig,
    SingleEvalBaselineMethod,
)
from src.utils import get_base_url


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a single LoRA on specified tasks.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
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

    config = SingleEvalBaselineConfig(
        tasks=args.tasks,
        test_tasks=args.test_tasks,
        task_weights=task_weights,
        model_name_or_path=args.model_path,
        llm_base_url=get_base_url(args.ports),
        pools=[args.lora_path],
        combine_method=args.combine_method,
        plot_enabled=args.plot_enabled,
        early_stop=args.early_stop,
        early_stop_iter=args.early_stop_iter,
        seed=args.seed,
        workspace_prefix=args.workspace_prefix,
        max_valid_samples=args.max_valid_samples,
    )

    method = SingleEvalBaselineMethod(config)
    method.search()


if __name__ == "__main__":
    main()
