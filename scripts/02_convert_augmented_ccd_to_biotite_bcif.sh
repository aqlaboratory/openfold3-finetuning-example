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
MERGED_CIF_DIR="${MERGED_CIF_DIR:-$DATASET_ROOT/merged_cifs}"
CCD_CIF="${CCD_CIF:-$MERGED_CIF_DIR/reference_data/components_harmonized.cif}"
CCD_BCIF="${CCD_BCIF:-$MERGED_CIF_DIR/reference_data/components_harmonized.bcif}"
SCRIPT_PATH="$REPO_ROOT/scripts/data_preprocessing/preprocess_ccd_biotite.py"

if [ ! -f "$CCD_CIF" ]; then
  echo "Augmented CCD CIF does not exist: $CCD_CIF" >&2
  echo "Run 01_merge_pdb_sdfs_to_cif.sh first." >&2
  exit 1
fi

cd "$REPO_ROOT"

python "$SCRIPT_PATH" "$CCD_CIF" "$CCD_BCIF"
