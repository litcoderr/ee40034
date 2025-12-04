#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="4"
RUN_NAME="${RUN_NAME:-baseline}"

# Data paths (override to point at your copies)
ROOT_PATH="/mnt/hard2/litcoderr/project/ee40034"
TRAIN_PATH="${TRAIN_PATH:-${ROOT_PATH}/data/train1}"
TEST_PATH="${TEST_PATH:-${ROOT_PATH}/data/val}"
TEST_LIST="${TEST_LIST:-${ROOT_PATH}/data/val_pairs.csv}"

# Training hyperparameters
MODEL="${MODEL:-ResNet18}"
TRAINFUNC="${TRAINFUNC:-softmax}"
BATCH_SIZE="${BATCH_SIZE:-512}"
N_CLASSES="${N_CLASSES:-2882}"   # set to 949 for train2
N_PER_CLASS="${N_PER_CLASS:-1}"
MAX_EPOCH="${MAX_EPOCH:-10}"
SAVE_ROOT="${SAVE_ROOT:-${SCRIPT_DIR}/exps}"

python "${SCRIPT_DIR}/trainEmbedNet.py" \
  --gpu "${GPU}" \
  --model "${MODEL}" \
  --trainfunc "${TRAINFUNC}" \
  --batch_size "${BATCH_SIZE}" \
  --nPerClass "${N_PER_CLASS}" \
  --nClasses "${N_CLASSES}" \
  --max_epoch "${MAX_EPOCH}" \
  --train_path "${TRAIN_PATH}" \
  --test_path "${TEST_PATH}" \
  --test_list "${TEST_LIST}" \
  --save_path "${SAVE_ROOT}/${RUN_NAME}" \
  --wandb_run_name "${RUN_NAME}" \
  "$@"
