#!/bin/bash
#SBATCH -J of3-finetune-pdb
#SBATCH -o examples/finetuning_2/slurm_outputs/finetune-pdb-%j.out
#SBATCH -e examples/finetuning_2/slurm_outputs/finetune-pdb-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=8:00:00

set -euo pipefail

# Avoid conda CUDA activate hooks failing under nounset when this var is unset.
: "${NVCC_PREPEND_FLAGS:=}"

module load gcc-native/12.3
module load cudatoolkit/24.11_12.6 

source ~/.bashrc
conda activate of3-internal

REPO_ROOT="/scratch/s5h/etowah.s5h/projects/openfold3-internal"
RUNNER_YAML=/scratch/s5h/etowah.s5h/projects/openfold3-internal/examples/finetuning_2/configs/finetune_two_node_155k.yaml

cd "${REPO_ROOT}"
mkdir -p examples/finetuning_2/slurm_outputs
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Optional: load site/compiler modules if needed on your cluster.
# module load gcc-native/12.3

# Optional caches to reduce rebuild overhead.
SCRATCH_DIR=/lus/lfs1aip2/scratch/s5h/etowah.s5h
export TRITON_HOME="${SCRATCH_DIR}/.triton"
export TRITON_CACHE_DIR="${TRITON_HOME}/cache"
export TORCH_EXTENSIONS_DIR="${SCRATCH_DIR}/.torch-extensions"

SEED="$(python -c 'import random; print(random.randint(0, 2**31 - 1))')"
echo "Starting training with seed: ${SEED}"
echo "Runner config: ${RUNNER_YAML}"

srun python -m openfold3.run_openfold train \
  --runner_yaml "${RUNNER_YAML}" \
  --seed "${SEED}"
