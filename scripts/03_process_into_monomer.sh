#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc

conda activate of3-internal

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENFOLD_REPO_ROOT="${OPENFOLD_REPO_ROOT:-${REPO_ROOT:-$FINETUNE_ROOT/../openfold3-internal}}"
REPO_ROOT="$OPENFOLD_REPO_ROOT"
export PYTHONPATH="$OPENFOLD_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
DATASET_ROOT="${DATASET_ROOT:-$FINETUNE_ROOT/example/dataset}"
MERGED_CIF_DIR="${MERGED_CIF_DIR:-$DATASET_ROOT/merged_cifs}"
MONOMER_CIF_DIR="${MONOMER_CIF_DIR:-$DATASET_ROOT/merged_cifs_monomer}"
MONOMER_LOG_PATH="${MONOMER_LOG_PATH:-$MONOMER_CIF_DIR/process_into_monomer.log}"
SCRIPT_PATH="$REPO_ROOT/examples/finetuning_2/dataset_generation/src/process_into_monomer.py"

if [ ! -d "$MERGED_CIF_DIR" ]; then
  echo "Merged CIF directory does not exist: $MERGED_CIF_DIR" >&2
  echo "Run 01_merge_pdb_sdfs_to_cif.sh first." >&2
  exit 1
fi

mkdir -p "$MONOMER_CIF_DIR"

cd "$REPO_ROOT"

python "$SCRIPT_PATH" \
  --input-dir "$MERGED_CIF_DIR" \
  --output-dir "$MONOMER_CIF_DIR" \
  --keep-only-main-ligand \
  --log-path "$MONOMER_LOG_PATH"
