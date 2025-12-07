#!/bin/bash

set -e

BATCH_SIZE=32
NAME="experiments_full"

# rm -rf .cache
# rm -rf ./$NAME
python training_full.py --output-base-dir ./$NAME --batch-size ${BATCH_SIZE} --task MeetingBank

python validate.py --validate-all ./$NAME --no-base-model