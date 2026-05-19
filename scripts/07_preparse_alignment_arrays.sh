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
ALIGNMENTS_DIR="${ALIGNMENTS_DIR:-$DATASET_ROOT/alignments}"
ALIGNMENTS_MAIN_DIR="${ALIGNMENTS_MAIN_DIR:-$ALIGNMENTS_DIR/main}"
ALIGNMENT_ARRAY_DIR="${ALIGNMENT_ARRAY_DIR:-$DATASET_ROOT/alignment_arrays}"
SCRIPT_PATH="$REPO_ROOT/scripts/data_preprocessing/preparse_alignments_of3.py"

if [ ! -d "$ALIGNMENTS_MAIN_DIR" ]; then
  echo "Raw alignments directory does not exist: $ALIGNMENTS_MAIN_DIR" >&2
  echo "Run 06_generate_colabfold_msas.sh first." >&2
  exit 1
fi

mkdir -p "$ALIGNMENT_ARRAY_DIR"

cd "$REPO_ROOT"

python "$SCRIPT_PATH" \
  --alignments_directory "$ALIGNMENTS_MAIN_DIR" \
  --alignment_array_directory "$ALIGNMENT_ARRAY_DIR" \
  --max_seq_counts '{"colabfold_main": 16384}' \
  --num_workers 4
