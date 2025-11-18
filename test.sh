#!/bin/bash

set -e

BATCH_SIZE=256
NAME="run_final_with_experience_replay"
# NAME="run_cstance_only"
# COMMON="--torch-compile"
# COMMON=""
# rm -rf .cache
# rm -rf ./$NAME
python training.py --output-base-dir ./$NAME --batch-size ${BATCH_SIZE}

python validate.py --validate-all ./$NAME/continual --no-base-model
# python validate.py --checkpoint-dir ./$NAME/continual/task_0_C-STANCE --batch-size ${BATCH_SIZE}
# python validate.py --checkpoint-dir ./$NAME/continual/task_0_FOMC --batch-size ${BATCH_SIZE}