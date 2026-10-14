#!/usr/bin/env bash
# Prepare the full OpenBind EV-A71 2A release (925 fragment-bound complexes) as
# OpenFold3 training data, then build the train and validation caches from the
# id lists next to this script.
#
# Steps:
#   1. Stage the Zenodo structures, merge PDB+SDF to mmCIF, build the augmented
#      CCD, monomerize, preprocess, and build alignment arrays from the MSA
#      shipped with the release
#   2. Train cache from train_ids.txt (79 structures)
#   3. Validation cache from val_ids.txt (64 structures)
#
# Training is NOT launched: the full dataset is normally trained on a multi-GPU
# node. See example/training/ for runner configs, and section 4 of the README.
#
# Usage (with the OpenFold3 conda env active):
#   ZENODO_DIR=/path/to/OpenBind_EV-A71_2A OPENFOLD_REPO_ROOT=/path/to/openfold3 \
#     ./run_ev_a71_2a_prep.sh
#
# Get the release from https://zenodo.org/records/22142262 and unzip it; it
# should contain structures/ and EV-A71_2A_cfdb_msa.a3m.
#
# Overrides:
#   ZENODO_DIR          unzipped release           (required)
#   OUTPUT_DIR          where outputs go           (default: <repo>/dataset/prepared)
#   OPENFOLD_REPO_ROOT  path to the OpenFold3 repo (required)
#   NUM_WORKERS         structure preprocessing    (default: 8)
#   DATASET_NAME        cache name prefix          (default: ev-a71-2a)
#   MSA                 provided | colabfold       (default: provided)
#   SKIP_PREP=1         reuse an existing OUTPUT_DIR and only rebuild caches
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -z "${OPENFOLD_REPO_ROOT:-}" ]]; then
  echo "OPENFOLD_REPO_ROOT is not set. Point it at your OpenFold3 repo, e.g." >&2
  echo "  OPENFOLD_REPO_ROOT=~/projects/openfold3 $0" >&2
  exit 1
fi
OUTPUT_DIR="${OUTPUT_DIR:-$FINETUNE_ROOT/dataset/prepared}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATASET_NAME="${DATASET_NAME:-ev-a71-2a}"
MSA="${MSA:-provided}"
PYTHON="${PYTHON:-python}"

if [[ -z "${ZENODO_DIR:-}" ]]; then
  echo "ZENODO_DIR is not set. Point it at the unzipped OpenBind_EV-A71_2A/ directory" >&2
  echo "from https://zenodo.org/records/22142262, e.g." >&2
  echo "  ZENODO_DIR=~/data/OpenBind_EV-A71_2A $0" >&2
  exit 1
fi
if [[ ! -d "$ZENODO_DIR/structures" ]]; then
  echo "No structures/ directory under $ZENODO_DIR -- is that the unzipped release root?" >&2
  exit 1
fi

# 1. Stage, merge to mmCIF, build the augmented CCD, monomerize, preprocess,
#    and build alignment arrays. MSA=provided crops the a3m shipped with the
#    release; MSA=colabfold queries the ColabFold server instead.
if [[ "${SKIP_PREP:-0}" != 1 ]]; then
  "$PYTHON" "$FINETUNE_ROOT/runners/prepare_zenodo_data.py" \
    --input_dir "$ZENODO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --openfold_repo "$OPENFOLD_REPO_ROOT" \
    --msa "$MSA" \
    --num_workers "$NUM_WORKERS"
fi

# 2. Train cache
"$PYTHON" "$FINETUNE_ROOT/runners/create_dataset_cache.py" train \
  --ids "$SCRIPT_DIR/train_ids.txt" \
  --dataset_dir "$OUTPUT_DIR" \
  --dataset_name "$DATASET_NAME-train" \
  --openfold_repo "$OPENFOLD_REPO_ROOT"

# 3. Validation cache
"$PYTHON" "$FINETUNE_ROOT/runners/create_dataset_cache.py" validation \
  --ids "$SCRIPT_DIR/val_ids.txt" \
  --dataset_dir "$OUTPUT_DIR" \
  --dataset_name "$DATASET_NAME-val" \
  --openfold_repo "$OPENFOLD_REPO_ROOT"

cat <<EOF

Done. Caches written to:
  $OUTPUT_DIR/caches/train_cache.json
  $OUTPUT_DIR/caches/val_cache.json

Inspect them before training (see "Train/validation caches" in the README), then
point a runner yaml from example/training/ at:
  alignment_array_directory:    $OUTPUT_DIR/alignment_arrays
  dataset_cache_file:           $OUTPUT_DIR/caches/train_cache.json
  target_structures_directory:  $OUTPUT_DIR/preprocessed/structure_files
  reference_molecule_directory: $OUTPUT_DIR/preprocessed/reference_mols
  ccd_file:                     $OUTPUT_DIR/merged_cifs/reference_data/components_harmonized.cif
EOF
