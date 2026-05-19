#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc

conda activate of3-internal

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENFOLD_REPO_ROOT="${OPENFOLD_REPO_ROOT:-${REPO_ROOT:-$FINETUNE_ROOT/../openfold3-internal}}"
REPO_ROOT="$OPENFOLD_REPO_ROOT"
export PYTHONPATH="$OPENFOLD_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
DATASET_ROOT="${DATASET_ROOT:-$FINETUNE_ROOT/example/dataset}"
INPUT_DIR="${INPUT_DIR:-$DATASET_ROOT/input_structures}"
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
