#!/bin/bash

set -e

BATCH_SIZE=64
NAME="experiments/run_20minuten"
TASKS_ARGS="--task 20Minuten"

# rm -rf .cache
# rm -rf ./$NAME
python -m src.training --output-base-dir ./${NAME} --data-dir TRACE-Benchmark/LLM-CL-Benchmark_5000 --batch-size ${BATCH_SIZE} $TASKS_ARGS

python -m src/validate --validate-all ./${NAME} --no-base-model ${TASKS_ARGS}