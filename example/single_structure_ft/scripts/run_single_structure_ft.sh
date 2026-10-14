#!/usr/bin/env bash
# Minimal fine-tuning example: train OpenFold3 on one EV-A71 2A structure (a71ev2a-x0812a)
# and validate on a close analog (a71ev2a-x2351a, the 5-fluoro version of the same ligand).
#
# Steps:
#   1. Preprocess dataset_raw/ (Zenodo layout) into dataset_processed/
#   2. Build the train and validation caches from dataset_raw/{train,val}_ids.txt
#   3. Write the runner yaml and launch training
#
# Usage (from any directory, with the OpenFold3 conda env active):
#   OPENFOLD_REPO_ROOT=/path/to/openfold3 ./run_single_structure_ft.sh
#
# Overrides:
#   OPENFOLD_REPO_ROOT  path to the OpenFold3 repo        (required)
#   CHECKPOINT          base model weights to fine-tune   (default: $OPENFOLD_REPO_ROOT/weights/of3-p2-155k.pt)
#   MAX_STEPS           number of optimizer steps         (default: 100)
#   VAL_EVERY           run validation every N steps      (default: 10)
#   SKIP_PREP=1         reuse an existing dataset_processed/
#   SKIP_TRAIN=1        only prepare the data
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FINETUNE_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"

if [[ -z "${OPENFOLD_REPO_ROOT:-}" ]]; then
  echo "OPENFOLD_REPO_ROOT is not set. Point it at your OpenFold3 repo, e.g." >&2
  echo "  OPENFOLD_REPO_ROOT=~/projects/openfold3 $0" >&2
  exit 1
fi
CHECKPOINT="${CHECKPOINT:-$OPENFOLD_REPO_ROOT/weights/of3-p2-155k.pt}"
MAX_STEPS="${MAX_STEPS:-100}"
VAL_EVERY="${VAL_EVERY:-10}"
PYTHON="${PYTHON:-python}"

RAW_DIR="$EXAMPLE_DIR/dataset_raw"
PROCESSED_DIR="$EXAMPLE_DIR/dataset_processed"
TRAIN_OUTPUT_DIR="$EXAMPLE_DIR/train_output"
RUNNER_YAML="$TRAIN_OUTPUT_DIR/runner.yml"

# 1. Stage structures, merge to mmCIF, monomerize, preprocess, and build alignment arrays
#    from the MSA shipped in dataset_raw/EV-A71_2A_cfdb_msa.a3m
if [[ "${SKIP_PREP:-0}" != 1 ]]; then
  "$PYTHON" "$FINETUNE_ROOT/runners/prepare_zenodo_data.py" \
    --input_dir "$RAW_DIR" \
    --output_dir "$PROCESSED_DIR" \
    --openfold_repo "$OPENFOLD_REPO_ROOT" \
    --msa provided
fi

# 2. Train cache (a71ev2a-x0812a) and validation cache (a71ev2a-x2351a)
"$PYTHON" "$FINETUNE_ROOT/runners/create_dataset_cache.py" train \
  --ids "$RAW_DIR/train_ids.txt" \
  --dataset_dir "$PROCESSED_DIR" \
  --dataset_name single-structure-ft-train \
  --openfold_repo "$OPENFOLD_REPO_ROOT"

"$PYTHON" "$FINETUNE_ROOT/runners/create_dataset_cache.py" validation \
  --ids "$RAW_DIR/val_ids.txt" \
  --dataset_dir "$PROCESSED_DIR" \
  --dataset_name single-structure-ft-val \
  --openfold_repo "$OPENFOLD_REPO_ROOT"

# 3. Runner yaml. With a single training structure, epoch_len sets how often validation runs.
mkdir -p "$TRAIN_OUTPUT_DIR"
cat > "$RUNNER_YAML" <<EOF
experiment_settings:
  mode: train
  output_dir: $TRAIN_OUTPUT_DIR
  restart_checkpoint_path: $CHECKPOINT
  ckpt_load_settings:
    manual_checkpoint_loading: true
    init_from_ema_weights: false
    restore_lr_scheduler: false
    restore_time_step: false
    strict_loading: false

data_module_args:
  batch_size: 1
  num_workers: 0
  epoch_len: $VAL_EVERY

logging_config:
  log_lr: false
  wandb_config: null

pl_trainer_args:
  devices: 1
  num_nodes: 1
  precision: bf16-mixed
  max_epochs: -1
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  log_every_n_steps: 1

checkpoint_config:
  every_n_epochs: 1
  auto_insert_metric_name: false
  save_last: true
  save_top_k: 0

model_update:
  presets:
    - train
  custom:
    architecture:
      shared:
        use_confidence_emb_prob: 0.8
        diffusion:
          use_conditioning_prob: 0.8
    settings:
      blocks_per_ckpt: 1

dataset_configs:
  train:
    weighted-pdb:
      dataset_class: WeightedPDBDataset
      weight: 1.0
      config:
        debug_mode: true
        template:
          n_templates: 4
        crop:
          token_crop:
            enabled: true
            token_budget: 384
            crop_weights:
              contiguous: 0.2
              spatial: 0.4
              spatial_interface: 0.4
          chain_crop:
            enabled: true

  validation:
    val-weighted-pdb:
      dataset_class: ValidationPDBDataset
      config:
        debug_mode: true
        msa:
          subsample_main: false
        template:
          n_templates: 4
        crop:
          token_crop:
            enabled: false

dataset_paths:
  weighted-pdb:
    alignments_directory: none
    alignment_db_directory: none
    alignment_array_directory: $PROCESSED_DIR/alignment_arrays
    dataset_cache_file: $PROCESSED_DIR/caches/train_cache.json
    target_structures_directory: $PROCESSED_DIR/preprocessed/structure_files
    target_structure_file_format: npz
    reference_molecule_directory: $PROCESSED_DIR/preprocessed/reference_mols
    template_cache_directory: none
    template_structure_array_directory: none
    template_structures_directory: none
    template_file_format: null
    ccd_file: null

  val-weighted-pdb:
    alignments_directory: none
    alignment_db_directory: none
    alignment_array_directory: $PROCESSED_DIR/alignment_arrays
    dataset_cache_file: $PROCESSED_DIR/caches/val_cache.json
    target_structures_directory: $PROCESSED_DIR/preprocessed/structure_files
    target_structure_file_format: npz
    reference_molecule_directory: $PROCESSED_DIR/preprocessed/reference_mols
    template_cache_directory: none
    template_structure_array_directory: none
    template_structures_directory: none
    template_file_format: null
    ccd_file: null
EOF
echo "Wrote runner config to $RUNNER_YAML"

if [[ "${SKIP_TRAIN:-0}" == 1 ]]; then
  echo "SKIP_TRAIN=1, not launching training. To train:"
  echo "  $PYTHON -m openfold3.run_openfold train --runner_yaml $RUNNER_YAML --seed 42"
  exit 0
fi

cd "$OPENFOLD_REPO_ROOT"
"$PYTHON" -m openfold3.run_openfold train --runner_yaml "$RUNNER_YAML" --seed 42
