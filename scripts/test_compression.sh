#!/bin/bash

set -e

NAME="experiments/run_final"
TASK_ARGS="--task MeetingBank --task FOMC --task Py150 --task ScienceQA --task NumGLUE-cm --task NumGLUE-ds --task 20Minuten"

python -m src.compare_merged_delta \
    ${TASK_ARGS} \
    --checkpoint-base-dir ./${NAME}/merged_models \
    --quantized-model-dir ./${NAME}/quantized_models \
    --comparison-mode both \
    --quantization 4bit