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
VAL_METADATA_CACHE="${VAL_METADATA_CACHE:-$PREPROCESSED_DIR/metadata_val.json}"
PREPROCESSED_STRUCTURE_DIR="${PREPROCESSED_STRUCTURE_DIR:-$PREPROCESSED_DIR/structure_files}"
ALIGNMENTS_DIR="${ALIGNMENTS_DIR:-$DATASET_ROOT/alignments}"
REPRESENTATIVES_FASTA="${REPRESENTATIVES_FASTA:-$ALIGNMENTS_DIR/alignment_representatives.fasta}"
ALIGNMENT_ARRAY_DIR="${ALIGNMENT_ARRAY_DIR:-$DATASET_ROOT/alignment_arrays}"
CACHE_DIR="${CACHE_DIR:-$DATASET_ROOT/caches}"
VAL_CACHE_OUTPUT="${VAL_CACHE_OUTPUT:-$CACHE_DIR/val_cache.json}"
VAL_DATASET_NAME="${VAL_DATASET_NAME:-a71ev2a-small-test-val}"
SCRIPT_PATH="$REPO_ROOT/examples/finetuning_2/dataset_generation/src/create_validation_cache.py"

if [ ! -f "$VAL_METADATA_CACHE" ]; then
  echo "Metadata cache does not exist: $VAL_METADATA_CACHE" >&2
  echo "Run 08_materialize_split_metadata.sh first." >&2
  exit 1
fi

if [ ! -d "$PREPROCESSED_STRUCTURE_DIR" ]; then
  echo "Preprocessed structure directory does not exist: $PREPROCESSED_STRUCTURE_DIR" >&2
  echo "Run 04_preprocess_openfold3_structures.sh first." >&2
  exit 1
fi

if [ ! -f "$REPRESENTATIVES_FASTA" ]; then
  echo "Alignment representatives FASTA does not exist: $REPRESENTATIVES_FASTA" >&2
  echo "Run 05_build_msa_representatives.sh first." >&2
  exit 1
fi

if [ ! -d "$ALIGNMENT_ARRAY_DIR" ]; then
  echo "Alignment arrays directory does not exist: $ALIGNMENT_ARRAY_DIR" >&2
  echo "Run 07_preparse_alignment_arrays.sh first." >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"

cd "$REPO_ROOT"

python "$SCRIPT_PATH" \
  --metadata-cache "$VAL_METADATA_CACHE" \
  --preprocessed-dir "$PREPROCESSED_STRUCTURE_DIR" \
  --alignment-representatives-fasta "$REPRESENTATIVES_FASTA" \
  --alignment-arrays-dir "$ALIGNMENT_ARRAY_DIR" \
  --output "$VAL_CACHE_OUTPUT" \
  --dataset-name "$VAL_DATASET_NAME"
