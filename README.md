# OpenFold3 finetuning example 

When using a cofolding model in a drug discovery campaign, the model should ideally reflect the
most up-to-date structural information available for the target — structures recently deposited in
the PDB, or experimental structures generated as the campaign progresses.

This repo contains the data-preparation pipeline and training configs used to fine-tune OpenFold3
on the first [OpenBind data release](https://zenodo.org/records/22142262) (79 fragment-bound
structures of the EV-A71 2A protease). Fine-tuning on those fragment structures improved pose
prediction and virtual-screening discrimination on related follow-on compounds, and works best
when the binding mode is conserved between the fragment and the follow-on compound.

The pipeline handles two cases:

- **Novel chemical matter** (`runners/prepare_zenodo_data.py`) — PDB + SDF pairs, needing a custom
  CCD and monomerization.
- **PDB entries** (`runners/prepare_pdb_data.py`) — mmCIF files that already use standard CCD codes.

Everything runs on anything from an 8-GPU node down to a single desktop GPU.

## Layout

| Path | What it is |
| --- | --- |
| `runners/` | Entry points. One command per pipeline stage. |
| `src/` | The individual preprocessing steps the runners call. |
| `example/single_structure_ft/` | End-to-end example: train on one structure, validate on a close analog. Includes raw and fully processed data. |
| `example/EV712a/` | The full 925-complex OpenBind EV-A71 2A dataset: a prep script plus the 79/64 train and validation id lists. |
| `example/training/` | Training configs for a 2-node run and for a single GPU with gradient accumulation. |

## Setup

Install this repo and OpenFold3 from source. The pipeline imports from the OpenFold3 repo and calls
its preprocessing scripts, so you need the source tree, not just an installed wheel.

```bash
git clone git@github.com:aqlaboratory/openfold3-finetuning-example.git

# Clone OF3 wherever you like; you point the runners at it explicitly
git clone git@github.com:aqlaboratory/openfold-3.git
cd openfold-3

conda create -n of3-internal python=3.13 -y
conda activate of3-internal

pip install -e .

setup_openfold
```

### Pointing at the OpenFold3 repo

Every runner needs to know where the OpenFold3 source tree is, and there is no default or
guessed location. Give it explicitly, either per invocation:

```bash
python runners/prepare_zenodo_data.py ... --openfold_repo /path/to/openfold3
```

or once for the shell, which the runners and the example scripts both read:

```bash
export OPENFOLD_REPO_ROOT=/path/to/openfold3
```

`--openfold_repo` wins over the environment variable. If neither is set, or the path does not
contain `scripts/data_preprocessing/`, the runner stops immediately and says so rather than failing
later inside a subprocess.

The runners use this to find `scripts/data_preprocessing/` (for `preprocess_ccd_biotite.py`,
`preprocess_pdb_of3.py` and `create_pdb-weighted_training_dataset_cache.py`) and to put the repo on
`PYTHONPATH` for the subprocesses they spawn. Run everything with the OpenFold3 conda env active —
the runners invoke `sys.executable`, so the subprocesses inherit the env you launch from.

### Model weights, and which OpenFold3 commit goes with which checkpoint

Fine-tuning starts from a base OpenFold3 checkpoint. **The base checkpoint determines which
OpenFold3 commit you must check out** — the two current checkpoints need different, mutually
exclusive versions, because the model architecture changed between them.

| Base checkpoint | File | OpenFold3 version | Check out |
| --- | --- | --- | --- |
| `openfold3-p2-155k` | `of3-p2-155k.pt` | `>=0.4,<0.4.4dev0` | tag `0.4.3` (`0bb17be5`) |
| `openbind-2025-06-30-174k` | `of3-ob-2025-06-30-174k.pt` | `>=0.5.0` | tag `v0.5.0` (`c4771653`) |

```bash
cd $OPENFOLD_REPO_ROOT

# to fine-tune openfold3-p2
git checkout 0.4.3

# to fine-tune openbind-0
git checkout v0.5.0

pip install -e .   # re-install after switching, the package version is pinned at install time
```

Those constraints are upstream's own, from `OPENFOLD_MODEL_CHECKPOINT_REGISTRY` in
`openfold3/entry_points/parameters.py` and `docs/source/parameters_reference.md` at `v0.5.0`.
Note the registry caps p2 at `<0.4.4dev0` while that doc table says `<0.5`; the registry is what
the code enforces, so `0.4.0`–`0.4.3` is the safe range for p2.

You cannot mix them. The two checkpoints are different architectures. p2 carries a per-block
`diffusion_transformer.blocks.N.attention_pair_bias.layer_norm_z` — 24 blocks, mirrored under
`sample_diffusion.`, so 48 parameters that openbind does not have (4,935 parameters in total).
openbind instead has one shared `diffusion_transformer.layer_norm_z` per copy (4,890). The openbind
checkpoint
also carries `version_tensor = [2, 0, 0]`, matching `MODEL_VERSION` at `v0.5.0`, whereas `0.4.0`
through `0.4.5` declare `[1, 0, 0]`. Loading the wrong pair fails with missing and unexpected
state-dict keys rather than silently misbehaving.

A legacy `openfold3-p1` (`of3_ft3_v1.pt`, `<0.4`) also exists but is deprecated upstream.

Everything in this repo was run and verified against tag `0.4.3` with `of3-p2-155k.pt`. The
preprocessing runners call into the OpenFold3 source tree, so if you switch to `v0.5.0` for
openbind, re-check the data preparation steps as well as training.

The example expects `$OPENFOLD_REPO_ROOT/weights/of3-p2-155k.pt`; override with
`CHECKPOINT=/path/to/weights.pt` or by editing `restart_checkpoint_path` in your runner yaml.
Running `setup_openfold` downloads the default checkpoint for whichever version is checked out.

## Quick start: the single-structure example

Before running anything at scale, confirm the whole path works end to end. This example trains on
one EV-A71 2A structure (`a71ev2a-x0812a`) and validates on a close analog (`a71ev2a-x2351a`, the
5-fluoro version of the same ligand), so you should see the model overfit quickly.

```bash
OPENFOLD_REPO_ROOT=/path/to/openfold3 ./example/single_structure_ft/scripts/run_single_structure_ft.sh
```

It preprocesses `dataset_raw/`, builds train and validation caches, writes a runner yaml, and
launches training. Overrides:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENFOLD_REPO_ROOT` | *(required)* | Path to the OpenFold3 repo |
| `CHECKPOINT` | `$OPENFOLD_REPO_ROOT/weights/of3-p2-155k.pt` | Base weights to fine-tune |
| `MAX_STEPS` | `100` | Optimizer steps |
| `VAL_EVERY` | `10` | Validate every N steps |
| `SKIP_PREP=1` | — | Reuse the existing `dataset_processed/` |
| `SKIP_TRAIN=1` | — | Only prepare data and write the yaml |

`example/single_structure_ft/dataset_processed/` is committed, so `SKIP_PREP=1` lets you go straight
to training, or compare your own preprocessing output against a known-good result.

## 1. Dataset preparation

### From the OpenBind Zenodo release

Download and unzip the [release](https://zenodo.org/records/22142262), then:

```bash
python runners/prepare_zenodo_data.py \
  --input_dir dataset/OpenBind_EV-A71_2A \
  --output_dir <dataset_dir>
```

To run this and the cache step below in one go, using the same train/validation
split as the write-up:

```bash
ZENODO_DIR=/path/to/OpenBind_EV-A71_2A OPENFOLD_REPO_ROOT=/path/to/openfold3 \
  ./example/EV712a/run_ev_a71_2a_prep.sh
```

Two steps here are specific to novel chemical matter and do not appear in typical PDB processing:

- **Monomerization.** The crystallized unit may be a dimer, but the protein functions as a monomer
  and virtual screening is done as a monomer. This takes the protein chain with the most
  protein-ligand contacts. (Typical PDB processing instead takes the first bioassembly.)
- **A custom CCD.** The chemical component dictionary specifies the connectivity of molecules in the
  training set. For PDB structures you can use the CCD the PDB provides, but novel ligands have to
  be added to it. The pipeline writes an extended CCD to
  `<dataset_dir>/merged_cifs/reference_data/components_harmonized.cif`.

Then the standard processing applies:

- **Sequence clustering** — identical sequences are grouped into one cluster, so data sampling works
  correctly and redundant MSAs are not generated.
- **MSA generation** — one MSA per unique sequence. By default (`--msa provided`) the MSA shipped
  with the release is cropped to each structure's sequence; `--msa colabfold` queries the ColabFoldDB
  API instead. For larger-scale MSA generation you'd want the sequence database locally.
- **Metadata cache** — everything is summarized into a single cache capturing the information
  relevant to each target.

Other flags: `--num_workers N` parallelizes the OpenFold3 structure preprocessing.

### From PDB mmCIF files

More typically you'll train on structures already in the PDB. You can find candidates by searching
a PDB-derived sequence database with MMseqs2 (which has GPU-accelerated variants) or another search
method, then point the runner at the folder of `.cif` files:

```bash
python runners/prepare_pdb_data.py \
  --input_dir <folder of .cif files> \
  --output_dir <dataset_dir>
```

No merging, synthetic CCD, or monomerization is needed. The PDB's own CCD is downloaded to
`<dataset_dir>/reference_data/components.cif` unless you pass `--ccd_path`. Pass `--msa_a3m` to crop
a precomputed MSA instead of querying ColabFold.

### Output layout

Both runners produce the same structure under `<dataset_dir>`, which the later steps depend on:

```
alignment_arrays/                       # preparsed MSAs, one .npz per representative sequence
alignments/
  alignment_representatives.fasta       # representative sequence per cluster
  main/<rep_id>/colabfold_main.a3m
merged_cifs/                            # Zenodo runner only
  reference_data/components_harmonized.{cif,bcif}
merged_cifs_monomer/                    # Zenodo runner only
preprocessed/
  metadata.json
  structure_files/<id>/<id>.{npz,cif,fasta}
  reference_mols/
```

## 2. Train/validation caches

With the dataset prepared, choose which subset is training and which is validation. Given a list of
ids per split, one per line:

```bash
python runners/create_dataset_cache.py train \
  --ids <train_ids.txt> --dataset_dir <dataset_dir>

python runners/create_dataset_cache.py validation \
  --ids <val_ids.txt> --dataset_dir <dataset_dir>
```

This writes `<dataset_dir>/caches/train_cache.json` and `val_cache.json`, which define the model's
training and validation datasets. Use `--output` and `--dataset_name` to override the defaults.

**Inspect the caches before training.** Check that clustering worked and that the chains you expect
are all present. Things to look for:

- The PDB occasionally has incorrectly annotated structures. We have seen the ligand of interest
  land outside the first bioassembly and get dropped during processing.
- For custom datasets, make sure the structural resolution field is set. Examples without a
  resolution value are ignored by the confidence head, so they never contribute to updating it.
- Caches can be subset afterwards. It's convenient to generate the union of every configuration you
  want to try, then narrow down per run.
- Build a single-structure train/val pair first and confirm you can overfit it. See
  [the example](#quick-start-the-single-structure-example).

## 3. Training yaml

The runner yaml points OpenFold3 at the caches and files you just generated:

```yaml
alignments_directory: none
alignment_db_directory: none
alignment_array_directory: {dataset_dir}/alignment_arrays
dataset_cache_file: {dataset_dir}/caches/train_cache.json
target_structures_directory: {dataset_dir}/preprocessed/structure_files
target_structure_file_format: npz
reference_molecule_directory: {dataset_dir}/preprocessed/reference_mols
template_cache_directory: none
template_structure_array_directory: {placeholder directory}
template_structures_directory: none
template_file_format: npz
ccd_file: {dataset_dir}/merged_cifs/reference_data/components_harmonized.cif
```

Set `ccd_file` to the custom CCD when training on novel chemical matter. For structures processed
from the PDB you can use the CCD file the PDB provides.

### Hyperparameters

The yaml is also where the hyperparameters that matter live. Starting points:

| Setting | Guidance |
| --- | --- |
| Nodes / GPUs per node | `pl_trainer_args.num_nodes` and `devices`, to match your machine. |
| Global batch size | `num_nodes × devices × batch_size`. Base OpenFold3 training uses 256, but for small homogeneous fine-tuning sets, 4 or 8 is sufficient. On a single GPU, use gradient accumulation (`manual_optimization.accumulate_grad_batches`). |
| Learning rate | **The most impactful knob.** `0.0003` works well at batch size 8 (anything in that vicinity). Re-tune it for a different batch size or a more heterogeneous dataset. |
| Crop size | `crop.token_crop.token_budget`. Base OpenFold3 uses 384, which is enough for protein-ligand fine-tuning. Every example is padded to the crop size, so if your dataset's maximum token count is below 384, lowering this gives significant memory and speed savings. |
| Warm-up steps | `lr_scheduler.warmup_no_steps`. 100–200 steps is sufficient. Full-parameter fine-tuning on a few structures fits very quickly, so some warm-up keeps metrics from saturating immediately. |
| EMA decay | OpenFold3 training uses `0.999` (half-life ~700 steps). Lowering it to ~`0.9` (half-life ~10 steps) makes the effect of fine-tuning visible much sooner. Unnecessary for low-rank fine-tuning. |

**Turn on the GPU kernels.** This dramatically reduces training memory and increases speed. Attention
kernels are enabled in the runner yaml via
`model_update.custom.settings.memory.train.use_deepspeed_evo_attention: true` (see
`example/training/single_gpu_gradient_accumulation.yml`). Also enable the CuEquivariance kernels for
the triangle operations — see the OpenFold3 repo for the current option name, since it is set on the
OpenFold3 side rather than here.

Two worked configs are in `example/training/`:

- `two_node_finetune.yaml` — 2 nodes × 4 GPUs, global batch size 8, lr `3e-4`, 200-step warm-up.
- `single_gpu_gradient_accumulation.yml` — 1 GPU with `accumulate_grad_batches: 8`.

Both contain absolute paths from the machines they were run on. Update `output_dir`,
`restart_checkpoint_path` and every path under `dataset_paths` before using them.

## 4. Start training

```bash
python -m openfold3.run_openfold train \
  --runner_yaml "${RUNNER_YAML}" \
  --seed "${SEED}"
```

On a Slurm system, prefix with `srun` to handle multi-node training:

```bash
srun python -m openfold3.run_openfold train \
  --runner_yaml "${RUNNER_YAML}" \
  --seed "${SEED}"
```

For reference, fine-tuning OpenFold3-p2 on the 79 fragment-bound structures for 1,000 steps at
batch size 8 with a 100-step linear warm-up takes roughly **8 hours on eight NVIDIA GH200 GPUs**, or
**50 hours on a single DGX Spark desktop GPU**. Expect this to vary with the target and the amount
of data, and to fall as GPU kernels improve.

Training produces a new OpenFold3 checkpoint that runs through the same inference pipeline as the
base model.

## What this means in practice

For this target, a small set of fragment-bound structures improved pose prediction and
virtual-screening discrimination on related follow-on compounds — and the fine-tuning could be done
on a single desktop GPU. A cofolding model does not have to stay fixed during a campaign: as new
structures are generated, they can be trained on and used to improve the next round of predictions.

We are still working out when fine-tuning helps. We expect it to help for some targets more than
others, and for some test entries more than others. We hope the workflow here is a useful starting
point for fine-tuning on your own data.

## Notes

- `merged_cifs/reference_data/` is gitignored. The CCD `components.cif` files are ~500 MB each,
  over GitHub's blob limit, and are regenerated by the preprocessing step.
