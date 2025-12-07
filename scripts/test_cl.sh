#!/bin/bash

set -e

BATCH_SIZE=64
NAME="experiments/run_final"
TASK_ARGS="--task MeetingBank --task FOMC --task Py150 --task ScienceQA --task NumGLUE-cm --task NumGLUE-ds --task 20Minuten"

# rm -rf .cache
# rm -rf ./$NAME
python src/training.py --output-base-dir ./${NAME} --data-dir TRACE-Benchmark/LLM-CL-Benchmark_5000 --batch-size ${BATCH_SIZE} ${TASK_ARGS}

python src/validate.py --validate-all ./${NAME} --no-base-model ${TASK_ARGS}