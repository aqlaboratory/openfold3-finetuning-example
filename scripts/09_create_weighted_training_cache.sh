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
TRAIN_METADATA_CACHE="${TRAIN_METADATA_CACHE:-$PREPROCESSED_DIR/metadata_train.json}"
PREPROCESSED_STRUCTURE_DIR="${PREPROCESSED_STRUCTURE_DIR:-$PREPROCESSED_DIR/structure_files}"
ALIGNMENTS_DIR="${ALIGNMENTS_DIR:-$DATASET_ROOT/alignments}"
REPRESENTATIVES_FASTA="${REPRESENTATIVES_FASTA:-$ALIGNMENTS_DIR/alignment_representatives.fasta}"
CACHE_DIR="${CACHE_DIR:-$DATASET_ROOT/caches}"
TRAIN_CACHE_OUTPUT="${TRAIN_CACHE_OUTPUT:-$CACHE_DIR/train_cache.json}"
MISSING_ALIGNMENT_LOG="${MISSING_ALIGNMENT_LOG:-$CACHE_DIR/missing_alignments.json}"
TRAIN_CACHE_LOG_FILE="${TRAIN_CACHE_LOG_FILE:-$CACHE_DIR/create_train_cache.log}"
TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-a71ev2a-small-test}"
SCRIPT_PATH="$REPO_ROOT/scripts/data_preprocessing/create_pdb-weighted_training_dataset_cache.py"

if [ ! -f "$TRAIN_METADATA_CACHE" ]; then
  echo "Metadata cache does not exist: $TRAIN_METADATA_CACHE" >&2
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

mkdir -p "$CACHE_DIR"

cd "$REPO_ROOT"

python "$SCRIPT_PATH" \
  --metadata-cache "$TRAIN_METADATA_CACHE" \
  --preprocessed-dir "$PREPROCESSED_STRUCTURE_DIR" \
  --alignment-representatives-fasta "$REPRESENTATIVES_FASTA" \
  --output "$TRAIN_CACHE_OUTPUT" \
  --dataset-name "$TRAIN_DATASET_NAME" \
  --missing-alignment-log "$MISSING_ALIGNMENT_LOG" \
  --log-level INFO \
  --log-file "$TRAIN_CACHE_LOG_FILE"
