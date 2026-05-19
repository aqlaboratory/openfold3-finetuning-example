#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc

# Avoid conda CUDA activate hooks failing under nounset when this var is unset.
: "${NVCC_PREPEND_FLAGS:=}"

conda activate of3-internal

REPO_ROOT="${REPO_ROOT:-/scratch/s5h/etowah.s5h/projects/openfold3-internal}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/a71ev2a/small_test}"
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
