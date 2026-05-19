#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc

# Avoid conda CUDA activate hooks failing under nounset when this var is unset.
: "${NVCC_PREPEND_FLAGS:=}"

conda activate of3-internal

REPO_ROOT="${REPO_ROOT:-/scratch/s5h/etowah.s5h/projects/openfold3-internal}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/datasets/a71ev2a/small_test}"
ALIGNMENTS_DIR="${ALIGNMENTS_DIR:-$DATASET_ROOT/alignments}"
REPRESENTATIVES_FASTA="${REPRESENTATIVES_FASTA:-$ALIGNMENTS_DIR/alignment_representatives.fasta}"
MSA_QUERY_DIR="${MSA_QUERY_DIR:-$DATASET_ROOT/msa_queries}"
MSA_GENERATION_REPORT_JSON="${MSA_GENERATION_REPORT_JSON:-$ALIGNMENTS_DIR/msa_generation_report.json}"
SCRIPT_PATH="$REPO_ROOT/examples/finetuning_2/dataset_generation/src/generate_msas.py"

if [ ! -f "$REPRESENTATIVES_FASTA" ]; then
  echo "Alignment representatives FASTA does not exist: $REPRESENTATIVES_FASTA" >&2
  echo "Run 05_build_msa_representatives.sh first." >&2
  exit 1
fi

cd "$REPO_ROOT"

python "$SCRIPT_PATH" \
  --representatives-fasta "$REPRESENTATIVES_FASTA" \
  --output-dir "$ALIGNMENTS_DIR" \
  --query-json-dir "$MSA_QUERY_DIR" \
  --report-json "$MSA_GENERATION_REPORT_JSON" \
  --log-level INFO
