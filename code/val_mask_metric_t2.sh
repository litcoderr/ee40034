#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="${GPU:-4}"
RUN_NAME="${RUN_NAME:-mask_triplet_arcface_t2_val}"

# Data paths
ROOT_PATH="/mnt/hard2/litcoderr/project/ee40034"
TRAIN_PATH="${TRAIN_PATH:-${ROOT_PATH}/data/train2}" # still needed for DataLoader init
TEST_PATH="${TEST_PATH:-${ROOT_PATH}/data/val}"
TEST_LIST="${TEST_LIST:-${ROOT_PATH}/data/val_pairs.csv}"

# Model/config (matches train_se_mask_metric_t2.sh)
MODEL="${MODEL:-resnet_mask}"
TRAINFUNC="${TRAINFUNC:-triplet_arcface}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_CLASSES="${N_CLASSES:-2882}"
N_PER_CLASS="${N_PER_CLASS:-2}"
N_OUT="${N_OUT:-256}"
SAVE_ROOT="${SAVE_ROOT:-${SCRIPT_DIR}/exps/mask_triplet_arcface_t2/eval}"
INITIAL_MODEL="${INITIAL_MODEL:-/mnt/hard2/litcoderr/project/ee40034/code/exps/mask_triplet_arcface_t2/epoch0022.model}"
N_ATTN_MAP="${N_ATTN_MAP:-0}"
ATTN_MAP_SAVE_PATH="${ATTN_MAP_SAVE_PATH:-${SAVE_ROOT}/${RUN_NAME}/attn_maps}"

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
  --initial_model "${INITIAL_MODEL}" \
  --wandb_run_name "${RUN_NAME}" \
  --arcface_weight 1.0 \
  --triplet_weight 1.0 \
  --eval \
  --attn_map \
  --attn_map_save_path "${ATTN_MAP_SAVE_PATH}" \
  --n_attn_map "${N_ATTN_MAP}" \
  "$@"
