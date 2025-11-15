# Merge3 LoRA Baseline Tutorial

This guide explains how the new Merge3-style baseline fits into GENOME, which components were added, and how to run it.

## Components

- `src/merge3/config.py` – `Merge3Config` extends `BaseConfig` and adds NSGA-II knobs (`nsga_generations`, `nsga_population`, `genotype_dimension`, `parent_sample_size`, `variable_bounds`). All other fields (`tasks`, `pools`, `combine_method`, etc.) reuse GENOME conventions.
- `src/merge3/merger.py` – `LoRAMerger` receives a genotype vector and parent LoRA paths, then calls `BaseMethod.merge_lora_weights` + `save_lora_weight` to write a new adapter inside the current workspace.
- `src/merge3/problem.py` – `Merge3Problem` samples parent adapters, materializes the merged adapter via `LoRAMerger`, wraps it into an `Individual`, and triggers the existing `BaseMethod.evaluate()` to obtain task and weighted scores.
- `src/merge3/method.py` – `Merge3LoRAMethod` subclasses `BaseMethod` and runs a simple generation/population loop. Each genotype is evaluated through `Merge3Problem`; best scores update the global state, and post-search testing uses `ensemble_test` like other baselines.
- `run_merge3.py` – CLI entry that mirrors `run_genome.py`: it parses model/task arguments, builds a `Merge3Config`, instantiates `Merge3LoRAMethod`, and calls `search()`.
- `scripts/merge3.sh` – Example shell runner that sets placeholders for model paths, tasks, weights, NSGA parameters, and iterates over multiple seeds.

## Workflow

1. **Parent Selection** – `Merge3Problem.sample_parents()` draws `parent_sample_size` adapters from the configured LoRA pools.
2. **Genotype → Adapter** – A genotype (continuous vector of length `genotype_dimension`) produces weights for each parent; `LoRAMerger.materialize()` merges those adapters using the chosen `combine_method` and stores the result in the workspace.
3. **Evaluation** – The merged adapter becomes an `Individual` evaluated on the tasks listed in the config via the existing OpenAI/vLLM evaluator stack. Weighted scores are fed back to the search loop.
4. **Search Loop** – `Merge3LoRAMethod.search()` repeats the above for `nsga_population` individuals per generation, across `nsga_generations`. Tracking/reporting relies on `BaseMethod` functions so logs/plots remain consistent with other methods.
5. **Testing** – After the final generation, the best adapter is evaluated on the test split through the standard `ensemble_test` call.

## Key Parameters

- `--nsga_generations`: Number of outer iterations (default 5).
- `--nsga_population`: Individuals evaluated per generation (default 4).
- `--genotype_dimension`: Length of the genotype vector; typically matches the number of merge weights you want to search.
- `--parent_sample_size`: How many LoRA parents are merged for each genotype.
- `--variable_bounds`: Lower/upper bounds for each genotype dimension.
- Standard GENOME arguments (`--tasks`, `--task_weights`, `--model_path`, `--lora_dir`, `--combine_method`, `--ports`, `--seed`) keep their original meaning.

## Running

```bash
bash scripts/merge3.sh
```

Before running, edit `scripts/merge3.sh` to point to real model and LoRA directories, adjust tasks/weights, and set hardware ports. Alternatively, call `python run_merge3.py ...` directly with your desired parameter values.
