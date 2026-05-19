#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc

# Avoid conda CUDA activate hooks failing under nounset when this var is unset.
: "${NVCC_PREPEND_FLAGS:=}"

conda activate of3-internal

REPO_ROOT="${REPO_ROOT:-/scratch/s5h/etowah.s5h/projects/openfold3-internal}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/a71ev2a/small_test}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-$DATASET_ROOT/preprocessed}"
METADATA_CACHE="${METADATA_CACHE:-$PREPROCESSED_DIR/metadata.json}"
SPLIT_DIR="${SPLIT_DIR:-$DATASET_ROOT/splits}"
TRAIN_IDS="${TRAIN_IDS:-$SPLIT_DIR/train_ids.txt}"
VAL_IDS="${VAL_IDS:-$SPLIT_DIR/val_ids.txt}"
TRAIN_METADATA_OUTPUT="${TRAIN_METADATA_OUTPUT:-$PREPROCESSED_DIR/metadata_train.json}"
VAL_METADATA_OUTPUT="${VAL_METADATA_OUTPUT:-$PREPROCESSED_DIR/metadata_val.json}"
SCRIPT_PATH="$REPO_ROOT/examples/finetuning_2/dataset_generation/src/materialize_split_metadata.py"

if [ ! -f "$METADATA_CACHE" ]; then
  echo "Metadata cache does not exist: $METADATA_CACHE" >&2
  echo "Run 04_preprocess_openfold3_structures.sh first." >&2
  exit 1
fi

if [ ! -f "$TRAIN_IDS" ]; then
  echo "Training id list does not exist: $TRAIN_IDS" >&2
  exit 1
fi

if [ ! -f "$VAL_IDS" ]; then
  echo "Validation id list does not exist: $VAL_IDS" >&2
  exit 1
fi

cd "$REPO_ROOT"

python "$SCRIPT_PATH" \
  --metadata-cache "$METADATA_CACHE" \
  --train-ids "$TRAIN_IDS" \
  --val-ids "$VAL_IDS" \
  --train-output "$TRAIN_METADATA_OUTPUT" \
  --val-output "$VAL_METADATA_OUTPUT"
