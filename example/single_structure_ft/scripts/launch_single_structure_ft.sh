#!/usr/bin/env bash
# The exact invocation used for the verified single-structure fine-tuning run.
#
# A thin wrapper over run_single_structure_ft.sh recording the machine-specific
# values that script expects to be told. The training hyperparameters are not
# here: they are the defaults in run_single_structure_ft.sh (32-true, EMA decay
# 0.9, 100 warmup steps, lr 3e-4) and a snapshot of the resulting config is in
# runner.yml next to this file.
#
# Usage:
#   ./launch_single_structure_ft.sh
#
# Edit the two paths below for your machine, or override them:
#   OPENFOLD_REPO_ROOT=... PYTHON=... ./launch_single_structure_ft.sh
#
# SKIP_PREP=1 is set because dataset_processed/ is committed. Drop it to
# regenerate the dataset from dataset_raw/ first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OPENFOLD_REPO_ROOT="${OPENFOLD_REPO_ROOT:-$HOME/projects/openfold3}"
export PYTHON="${PYTHON:-$HOME/miniconda3/envs/of3p2/bin/python}"
export SKIP_PREP="${SKIP_PREP:-1}"
export WANDB_NAME="${WANDB_NAME:-single-structure-ft}"

exec "$SCRIPT_DIR/run_single_structure_ft.sh"
