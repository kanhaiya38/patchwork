NAME="experiments/run_final"
TASK_ARGS="--task MeetingBank --task FOMC --task Py150 --task ScienceQA --task NumGLUE-cm --task NumGLUE-ds --task 20Minuten"

python src/compare_merged_delta.py \
    ${TASK_ARGS} \
    --checkpoint-base-dir ./${NAME}/merged_models \
    --comparison-mode both \
    --quantization all