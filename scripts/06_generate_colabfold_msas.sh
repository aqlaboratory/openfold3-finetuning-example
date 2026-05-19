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
