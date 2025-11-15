#!/usr/bin/env bash

export TOKENIZERS_PARALLELISM=true

MODEL_PATH="input your model name here"
LORA_PATH="input your lora directory here"
TASKS="mmlu gsm8k"
TEST_TASKS="mmlu gsm8k"
WEIGHTS="0.5 0.5"
COMBINE_METHOD=ties
NSGA_GENERATIONS=5
NSGA_POPULATION=4
GENOTYPE_DIM=2
PARENT_SAMPLE_SIZE=2
PORTS=(18177 36048 22246 13732)
SEEDS=(41 42)

for SEED in "${SEEDS[@]}"; do
    echo "Running Merge3 baseline with seed: $SEED"
    python run_merge3.py \
        --tasks $TASKS \
        --test_tasks $TEST_TASKS \
        --task_weights $WEIGHTS \
        --model_path $MODEL_PATH \
        --lora_dir $LORA_PATH \
        --combine_method $COMBINE_METHOD \
        --nsga_generations $NSGA_GENERATIONS \
        --nsga_population $NSGA_POPULATION \
        --genotype_dimension $GENOTYPE_DIM \
        --parent_sample_size $PARENT_SAMPLE_SIZE \
        --variable_bounds 0.0 1.0 \
        --ports "${PORTS[@]}" \
        --seed $SEED \
        --plot_enabled

    echo "Seed $SEED finished, sleeping 10 seconds"
    sleep 10
done
