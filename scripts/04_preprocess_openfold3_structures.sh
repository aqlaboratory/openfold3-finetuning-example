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
MONOMER_CIF_DIR="${MONOMER_CIF_DIR:-$DATASET_ROOT/merged_cifs_monomer}"
CCD_CIF="${CCD_CIF:-$MERGED_CIF_DIR/reference_data/components_harmonized.cif}"
CCD_BCIF="${CCD_BCIF:-$MERGED_CIF_DIR/reference_data/components_harmonized.bcif}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-$DATASET_ROOT/preprocessed}"
SCRIPT_PATH="$REPO_ROOT/scripts/data_preprocessing/preprocess_pdb_of3.py"

if [ ! -d "$MONOMER_CIF_DIR" ]; then
  echo "Monomerized CIF directory does not exist: $MONOMER_CIF_DIR" >&2
  echo "Run 03_process_into_monomer.sh first." >&2
  exit 1
fi

if [ ! -f "$CCD_CIF" ]; then
  echo "Augmented CCD CIF does not exist: $CCD_CIF" >&2
  echo "Run 01_merge_pdb_sdfs_to_cif.sh first." >&2
  exit 1
fi

if [ ! -f "$CCD_BCIF" ]; then
  echo "Augmented CCD BinaryCIF does not exist: $CCD_BCIF" >&2
  echo "Run 02_convert_augmented_ccd_to_biotite_bcif.sh first." >&2
  exit 1
fi

mkdir -p "$PREPROCESSED_DIR"

cd "$REPO_ROOT"

python "$SCRIPT_PATH" \
  --cif-dir "$MONOMER_CIF_DIR" \
  --ccd-path "$CCD_CIF" \
  --biotite-ccd-path "$CCD_BCIF" \
  --out-dir "$PREPROCESSED_DIR" \
  --output-format npz \
  --output-format cif \
  --num-workers 0 \
  --log-level INFO
