#!/bin/bash

set -e

BATCH_SIZE=32
NAME="experiments_full"

# rm -rf .cache
# rm -rf ./$NAME
python -m src.training_full --output-base-dir ./$NAME --batch-size ${BATCH_SIZE} --task MeetingBank

python -m src.validate --validate-all ./$NAME --no-base-model