#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc

# Avoid conda CUDA activate hooks failing under nounset when this var is unset.
: "${NVCC_PREPEND_FLAGS:=}"

conda activate of3-internal

REPO_ROOT="${REPO_ROOT:-/scratch/s5h/etowah.s5h/projects/openfold3-internal}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/a71ev2a/small_test}"
INPUT_DIR="${INPUT_DIR:-$DATASET_ROOT/example_input_small}"
MERGED_CIF_DIR="${MERGED_CIF_DIR:-$DATASET_ROOT/merged_cifs}"
SCRIPT_PATH="$REPO_ROOT/examples/finetuning_2/dataset_generation/src/merge_pdb_sdfs_to_cif.py"

if [ ! -d "$INPUT_DIR" ]; then
  echo "Input directory does not exist: $INPUT_DIR" >&2
  exit 1
fi

mkdir -p "$MERGED_CIF_DIR"

cd "$REPO_ROOT"

python "$SCRIPT_PATH" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$MERGED_CIF_DIR"
