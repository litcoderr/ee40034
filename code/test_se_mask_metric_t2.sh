#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="${GPU:-4}"
RUN_NAME="${RUN_NAME:-se_mask_triplet_arcface_t2_eval}"

# Data paths
ROOT_PATH="/mnt/hard2/litcoderr/project/ee40034"
TRAIN_PATH="${TRAIN_PATH:-${ROOT_PATH}/data/train2}"
TEST_PATH="${TEST_PATH:-${ROOT_PATH}/data/test}"
TEST_LIST="${TEST_LIST:-${ROOT_PATH}/data/test_pairs.csv}"

# Model/config (matches train_se_mask_metric_t2.sh)
MODEL="${MODEL:-resnet_se_mask}"
TRAINFUNC="${TRAINFUNC:-triplet_arcface}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_CLASSES="${N_CLASSES:-2882}"
N_PER_CLASS="${N_PER_CLASS:-2}"
N_OUT="${N_OUT:-256}"
SAVE_ROOT="${SAVE_ROOT:-${SCRIPT_DIR}/exps/se_mask_triplet_arcface_t2/eval}"
INITIAL_MODEL="${INITIAL_MODEL:-/mnt/hard2/litcoderr/project/ee40034/code/exps/se_mask_triplet_arcface_t2/epoch0028.model}"

# Disable W&B during evaluation
WANDB_MODE="${WANDB_MODE:-disabled}" \
python "${SCRIPT_DIR}/trainEmbedNet.py" \
  --gpu "${GPU}" \
  --model "${MODEL}" \
  --trainfunc "${TRAINFUNC}" \
  --batch_size "${BATCH_SIZE}" \
  --nOut "${N_OUT}" \
  --nPerClass "${N_PER_CLASS}" \
  --nClasses "${N_CLASSES}" \
  --train_path "${TRAIN_PATH}" \
  --test_path "${TEST_PATH}" \
  --test_list "${TEST_LIST}" \
  --save_path "${SAVE_ROOT}/${RUN_NAME}" \
  --output "${SAVE_ROOT}/${RUN_NAME}/test_output.csv" \
  --initial_model "${INITIAL_MODEL}" \
  --wandb_run_name "${RUN_NAME}" \
  --eval \
  "$@"
