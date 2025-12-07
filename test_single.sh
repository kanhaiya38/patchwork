#!/bin/bash

set -e

BATCH_SIZE=64
NAME="experiments/run_20minuten"
TASKS_ARGS="--task 20Minuten"
# NAME="run_final_meetingbank"
# NAME="run_cstance_only"
# COMMON="--torch-compile"
# COMMON=""
# rm -rf .cache
# rm -rf ./$NAME
# python training.py --output-base-dir ./$NAME --data-dir TRACE-Benchmark/LLM-CL-Benchmark_5000 --batch-size ${BATCH_SIZE} $TASKS_ARGS

python validate.py --validate-all ./$NAME/continual --no-base-model $TASKS_ARGS
# python validate.py --checkpoint-dir ./$NAME/continual/task_0_C-STANCE --batch-size ${BATCH_SIZE}
# python validate.py --checkpoint-dir ./$NAME/continual/task_0_FOMC --batch-size ${BATCH_SIZE}