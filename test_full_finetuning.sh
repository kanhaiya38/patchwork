#!/bin/bash

set -e

BATCH_SIZE=32
NAME="experiments_full"
# NAME="run_cstance_only"
# COMMON="--torch-compile"
# COMMON=""
# rm -rf .cache
rm -rf ./$NAME
python training_full.py --output-base-dir ./$NAME --batch-size ${BATCH_SIZE} --task MeetingBank

python validate.py --validate-all ./$NAME/continual --no-base-model
# python validate.py --checkpoint-dir ./$NAME/continual/task_0_C-STANCE --batch-size ${BATCH_SIZE}
# python validate.py --checkpoint-dir ./$NAME/continual/task_0_FOMC --batch-size ${BATCH_SIZE}
# python validate.py --checkpoint-dir ./$NAME/continual/task_0_MeetingBank --batch-size ${BATCH_SIZE}