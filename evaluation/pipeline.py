from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional


class EvaluationPipeline:
    """Composable evaluation pipeline with hook support."""

    def __init__(
        self,
        prepare_input: Optional[Callable[[Any, str, Dict[str, Any]], Dict[str, Any]]] = None,
        run_inference: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Any]] = None,
        score: Optional[Callable[[Any, Dict[str, Any]], Any]] = None,
        post_process: Optional[Callable[[str, Any, Dict[str, Any]], Any]] = None,
    ) -> None:
        self.prepare_input = prepare_input
        self.run_inference = run_inference
        self.score = score
        self.post_process = post_process
        self._hooks: Dict[str, List[Callable[..., None]]] = defaultdict(list)

    def configure(
        self,
        *,
        prepare_input: Callable[[Any, str, Dict[str, Any]], Dict[str, Any]],
        run_inference: Callable[[Dict[str, Any], Dict[str, Any]], Any],
        score: Callable[[Any, Dict[str, Any]], Any],
        post_process: Callable[[str, Any, Dict[str, Any]], Any],
    ) -> None:
        """Configure the pipeline steps."""

        self.prepare_input = prepare_input
        self.run_inference = run_inference
        self.score = score
        self.post_process = post_process

    def register_hook(self, event: str, callback: Callable[..., None]) -> None:
        """Register a hook for a specific pipeline event."""

        self._hooks[event].append(callback)

    def _trigger(self, event: str, **payload: Any) -> None:
        for hook in self._hooks.get(event, []):
            hook(**payload)

    def run(
        self,
        individual: Any,
        tasks: Iterable[str],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the evaluation pipeline for the provided tasks."""

        if not all([self.prepare_input, self.run_inference, self.score, self.post_process]):
            raise RuntimeError("Pipeline steps must be configured before running.")

        context = dict(context or {})
        context.setdefault("results", {})

        tasks_list = list(tasks)
        self._trigger(
            "before_pipeline",
            individual=individual,
            tasks=tasks_list,
            context=context,
        )

        results: Dict[str, Any] = {}
        for task in tasks_list:
            context["current_task"] = task
            self._trigger("before_task", individual=individual, task=task, context=context)

            prepared = self.prepare_input(individual, task, context)
            self._trigger(
                "after_prepare_input",
                individual=individual,
                task=task,
                context=context,
                data=prepared,
            )

            inference_output = self.run_inference(prepared, context)
            self._trigger(
                "after_run_inference",
                individual=individual,
                task=task,
                context=context,
                data=inference_output,
            )

            scored = self.score(inference_output, context)
            self._trigger(
                "after_score",
                individual=individual,
                task=task,
                context=context,
                data=scored,
            )

            processed = self.post_process(task, scored, context)
            self._trigger(
                "after_post_process",
                individual=individual,
                task=task,
                context=context,
                data=processed,
            )

            results[task] = processed
            context["results"][task] = processed

            self._trigger(
                "after_task",
                individual=individual,
                task=task,
                context=context,
                data=processed,
            )

        self._trigger(
            "after_pipeline",
            individual=individual,
            tasks=tasks_list,
            context=context,
            data=results,
        )

        context.pop("current_task", None)
        return results
