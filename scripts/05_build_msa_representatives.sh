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
PREPROCESSED_DIR="${PREPROCESSED_DIR:-$DATASET_ROOT/preprocessed}"
PREPROCESSED_STRUCTURE_DIR="${PREPROCESSED_STRUCTURE_DIR:-$PREPROCESSED_DIR/structure_files}"
ALIGNMENTS_DIR="${ALIGNMENTS_DIR:-$DATASET_ROOT/alignments}"
REPRESENTATIVES_FASTA="${REPRESENTATIVES_FASTA:-$ALIGNMENTS_DIR/alignment_representatives.fasta}"
ALIGNMENT_REPRESENTATIVES_REPORT_JSON="${ALIGNMENT_REPRESENTATIVES_REPORT_JSON:-$ALIGNMENTS_DIR/alignment_representatives_report.json}"
SCRIPT_PATH="$REPO_ROOT/examples/finetuning_2/dataset_generation/src/build_alignment_representatives.py"

if [ ! -d "$PREPROCESSED_STRUCTURE_DIR" ]; then
  echo "Preprocessed structure directory does not exist: $PREPROCESSED_STRUCTURE_DIR" >&2
  echo "Run 04_preprocess_openfold3_structures.sh first." >&2
  exit 1
fi

mkdir -p "$(dirname "$REPRESENTATIVES_FASTA")"

cd "$REPO_ROOT"

python "$SCRIPT_PATH" \
  --preprocessed-structure-dir "$PREPROCESSED_STRUCTURE_DIR" \
  --output-fasta "$REPRESENTATIVES_FASTA" \
  --report-json "$ALIGNMENT_REPRESENTATIVES_REPORT_JSON"
