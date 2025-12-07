#!/bin/bash

set -e

# Modify this if you encounter memory issues
BATCH_SIZE=64
NAME="experiments/run_final"
TASK_ARGS="--task MeetingBank --task FOMC --task Py150 --task ScienceQA --task NumGLUE-cm --task NumGLUE-ds --task 20Minuten"

# rm -rf .cache
# rm -rf ./$NAME
python -m src.training --output-base-dir ./${NAME} --data-dir TRACE-Benchmark/LLM-CL-Benchmark_5000 --batch-size ${BATCH_SIZE} ${TASK_ARGS}

# Validate full-precision model (Optional)
# python -m src.validate --validate-all ./${NAME} --no-base-model ${TASK_ARGS}

# Quantize models
python -m src.quantize_models --input-dir ./${NAME}/merged_models --output-dir ./${NAME}/quantized_models --quantization 4

# Validate quantized models
python -m src.validate --validate-all ./${NAME}/quantized_models --no-base-model --pre-quantized ${TASK_ARGS}