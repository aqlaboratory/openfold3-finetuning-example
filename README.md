# Finetuning Dataset Generation

## Setup

Install this repo and OpenFold3 from source. This script relies on imports from the OpenFold3 repo. 
```
git clone git@github.com:etowahadams/openfold3-finetune.git
# clone OF3
git clone git@github.com:aqlaboratory/openfold-3.git
cd openfold-3

conda create -n of3-internal python=3.13 -y
conda activate of3-internal

pip install -e .

setup_openfold
```

## Running the pipeline 

This pipeline starts from Fragalysis-style sample directories containing PDB and
SDF files, then produces OpenFold3-ready structure files, alignment arrays, a
weighted training dataset cache, and a full validation cache. It is
intentionally split into small, inspectable steps so each output can be checked
before moving on.


```bash
FINETUNE_ROOT="/path/to/openfold3-finetune"
DATASET_ROOT="$FINETUNE_ROOT/example/dataset"
OPENFOLD_REPO_ROOT="/path/to/openfold3-internal"
```

The runnable wrappers in this repo read those defaults from `scripts/all.sh`.
Set `OPENFOLD_REPO_ROOT` there, or override it at runtime:

```bash
OPENFOLD_REPO_ROOT="/path/to/openfold3-internal" ./scripts/all.sh
```

By default, the step scripts use `$FINETUNE_ROOT/example/dataset` for data and
`$FINETUNE_ROOT/../openfold3-internal` for OpenFold3 code.

## Workflow

### 1. Merge PDB/SDF Samples Into mmCIF

This step 1 will probably need to be custom written for your particular input format. In OpenBind, our data comes from a database called Fragalysis. We need to convert the this format into a format that is compatible with downstream scripts. 

We write a custom script which turns the Fragalysis format into a  `merge_pdb_sdfs_to_cif.py` on the Fragalysis-style input directory. This
creates one merged `.cif` file per sample and writes an augmented CCD containing
any promoted ligands used by those CIFs.

```bash
python examples/finetuning_2/dataset_generation/src/merge_pdb_sdfs_to_cif.py \
  --input-dir "$DATASET_ROOT/input_structures" \
  --output-dir "$DATASET_ROOT/merged_cifs"
```

Expected important outputs:

```text
$DATASET_ROOT/merged_cifs/*.cif
$DATASET_ROOT/merged_cifs/reference_data/components_harmonized.cif
$DATASET_ROOT/merged_cifs/merge_reports/
```

Use the augmented CCD from this step for all downstream structure preprocessing.
Do not accidentally switch back to the base `components.cif` if the merged CIFs
contain promoted ligand component IDs.

### 2. Convert the Augmented CCD to Biotite BinaryCIF

Create the matching `.bcif` file for Biotite from the augmented CCD produced in
step 1.

```bash
python scripts/data_preprocessing/preprocess_ccd_biotite.py \
  "$DATASET_ROOT/merged_cifs/reference_data/components_harmonized.cif" \
  "$DATASET_ROOT/merged_cifs/reference_data/components_harmonized.bcif"
```

The `.cif` CCD and `.bcif` CCD should represent the same augmented CCD snapshot.

### 3. Process Merged CIFs Into Monomers

Before OpenFold3 preprocessing, filter each merged CIF down to a single protein
chain plus the relevant ligand context. This step should:

- require exactly one multi-atom hetero chain that is not an ion, water, or `DMS`
- keep the protein chain closest to that main ligand chain
- always keep the main ligand chain
- keep any other hetero residues with at least one atom within 6 A of the
  selected protein chain

Run:

```bash
python examples/finetuning_2/dataset_generation/src/process_into_monomer.py \
  --input-dir "$DATASET_ROOT/merged_cifs" \
  --output-dir "$DATASET_ROOT/merged_cifs_monomer" \
  --log-path "$DATASET_ROOT/merged_cifs_monomer/process_into_monomer.log"
```

Expected important outputs:

```text
$DATASET_ROOT/merged_cifs_monomer/*.cif
$DATASET_ROOT/merged_cifs_monomer/process_into_monomer.log
```

The small-test workflow wraps this command in:

```bash
scripts/03_process_into_monomer.sh
```

### 4. Run OpenFold3 Structure Preprocessing

Run `preprocess_pdb_of3.py` on the monomerized CIFs from step 3. Pass both the
augmented `.cif` CCD and the matching `.bcif` CCD from step 2.

```bash
python scripts/data_preprocessing/preprocess_pdb_of3.py \
  --cif-dir "$DATASET_ROOT/merged_cifs_monomer" \
  --ccd-path "$DATASET_ROOT/merged_cifs/reference_data/components_harmonized.cif" \
  --biotite-ccd-path "$DATASET_ROOT/merged_cifs/reference_data/components_harmonized.bcif" \
  --out-dir "$DATASET_ROOT/preprocessed" \
  --output-format npz \
  --output-format cif \
  --num-workers 0 \
  --log-level INFO
```

Expected important outputs:

```text
$DATASET_ROOT/preprocessed/metadata.json
$DATASET_ROOT/preprocessed/structure_files/<structure_id>/<structure_id>.npz
$DATASET_ROOT/preprocessed/structure_files/<structure_id>/<structure_id>.fasta
$DATASET_ROOT/preprocessed/reference_mols/
```

The preprocessed FASTA files are the source of truth for MSA representative
generation.

### 5. Build MSA Representatives

Build the alignment representatives FASTA from the preprocessed FASTA files in
`structure_files`, not from the raw merged CIFs. The cache creator later maps
chains to alignments by comparing these same preprocessed sequences.

The representative IDs should match the MSA directory names and final NPZ names:

```text
alignments/alignment_representatives.fasta
alignments/main/<rep_id>/colabfold_main.a3m
alignment_arrays/<rep_id>.npz
```

The planned helper script should write:

```text
$DATASET_ROOT/alignments/alignment_representatives.fasta
```

### 6. Generate ColabFold Server MSAs

Use `generate_msas.py` to query the ColabFold MSA server for each representative
sequence. The output must be compatible with `preparse_alignments_of3.py`.

Required raw alignment layout:

```text
$DATASET_ROOT/alignments/main/<rep_id>/colabfold_main.a3m
```

Example intended command:

```bash
python examples/finetuning_2/dataset_generation/src/generate_msas.py \
  --representatives-fasta "$DATASET_ROOT/alignments/alignment_representatives.fasta" \
  --output-dir "$DATASET_ROOT/alignments" \
  --query-json-dir "$DATASET_ROOT/msa_queries"
```

If a generated A3M already exists, the script should verify that the returned
query sequence column count matches the submitted representative sequence before
reusing it.

### 7. Preparse Raw MSAs Into Alignment Arrays

Convert the raw `colabfold_main.a3m` files into compressed per-representative
NPZ files.

```bash
mkdir -p "$DATASET_ROOT/alignment_arrays"

python scripts/data_preprocessing/preparse_alignments_of3.py \
  --alignments_directory "$DATASET_ROOT/alignments/main" \
  --alignment_array_directory "$DATASET_ROOT/alignment_arrays" \
  --max_seq_counts '{"colabfold_main": 16384}' \
  --num_workers 4
```

Expected output:

```text
$DATASET_ROOT/alignment_arrays/<rep_id>.npz
```

`preparse_alignments_of3.py` expects each immediate child of
`alignments/main` to be one representative directory.

### 8. Materialize Split Metadata Files

Once the alignment arrays exist, split the original preprocessing
`metadata.json` into `metadata_train.json` and `metadata_val.json` using
explicit structure-id lists. This keeps the train/validation split local to the
finetuning pipeline instead of requiring separate preprocessing runs.

Add a helper script that reads two text files with one structure id per line:

```text
$DATASET_ROOT/input_splits/train_ids.txt
$DATASET_ROOT/input_splits/val_ids.txt
```

Then materialize the split metadata files with:

```bash
python examples/finetuning_2/dataset_generation/src/materialize_split_metadata.py \
  --metadata-cache "$DATASET_ROOT/preprocessed/metadata.json" \
  --train-ids "$DATASET_ROOT/input_splits/train_ids.txt" \
  --val-ids "$DATASET_ROOT/input_splits/val_ids.txt" \
  --train-output "$DATASET_ROOT/preprocessed/metadata_train.json" \
  --val-output "$DATASET_ROOT/preprocessed/metadata_val.json"
```

Expected outputs:

```text
$DATASET_ROOT/preprocessed/metadata_train.json
$DATASET_ROOT/preprocessed/metadata_val.json
```

The script should:

- fail if either id list is empty
- fail if an id appears in both files
- fail if any requested id is missing from the original `metadata.json`
- keep only the selected `structure_data` entries in each output file
- carry along the referenced `reference_molecule_data` entries needed by those structures

The small-test workflow wraps this command in:

```bash
scripts/08_materialize_split_metadata.sh
```

### 9. Create the Weighted Training Dataset Cache

Use the training metadata cache from step 8, the preprocessed
`structure_files` directory, and the alignment representatives FASTA.

```bash
mkdir -p "$DATASET_ROOT/caches"

python scripts/data_preprocessing/create_pdb-weighted_training_dataset_cache.py \
  --metadata-cache "$DATASET_ROOT/preprocessed/metadata_train.json" \
  --preprocessed-dir "$DATASET_ROOT/preprocessed/structure_files" \
  --alignment-representatives-fasta "$DATASET_ROOT/alignments/alignment_representatives.fasta" \
  --output "$DATASET_ROOT/caches/train_cache.json" \
  --dataset-name "a71ev2a-small-test" \
  --allow-missing-alignment \
  --missing-alignment-log "$DATASET_ROOT/caches/missing_alignments.json" \
  --log-level INFO
```

This script requires `mmseqs` for protein clustering. It is available in the
`of3-internal` conda environment, so run this command only after activating that
environment.

### 10. Create a Full Validation Cache

The stock PDB validation script,
`scripts/data_preprocessing/create_pdb_validation_dataset_cache.py`, should not
be used for the finetuning validation split when the goal is to evaluate on
every validation structure. That script implements the AF3/PDB validation set
construction: date filtering, homology labeling against training caches,
ligand-quality filtering, monomer/multimer subsampling, and final metric
selection. For this workflow, write a simpler script named
`create_validation_cache.py` that converts the validation metadata cache
directly into a `ValidationDatasetCache`.


Required inputs:

- `--metadata-cache`: the validation `metadata_val.json` produced by the step-8
  split helper.
- `--preprocessed-dir`: the shared `structure_files` directory containing
  `<structure_id>/<structure_id>.fasta` files.
- `--alignment-representatives-fasta`: the representatives FASTA created from
  the preprocessed FASTAs.
- `--output`: destination `val_cache.json`.
- `--dataset-name`: name stored in the cache.

Recommended optional input:

- `--alignment-arrays-dir`: directory containing `<rep_id>.npz` files. The
  script does not need these files to write the cache, but validating their
  presence catches missing MSA preprocessing before training or evaluation.

Implementation outline:

1. Load the metadata cache with `PreprocessingDataCache.from_json`.
2. Fail if any structure in the metadata cache is not a successful
   preprocessing result, or if its structure NPZ/FASTA is missing. Do not
   silently filter these entries, because the purpose of this cache is to
   contain everything in the metadata cache.
3. Convert the metadata cache with
   `build_provisional_clustered_val_dataset_cache`.
4. Read preprocessed chain sequences with `consolidate_preprocessed_fastas`.
5. Read representative sequences with `get_chain_id_to_seq_from_fasta`.
6. Assign `alignment_representative_id` values with `add_chain_representatives`.
7. Fail if any protein or RNA chain lacks an alignment representative. This is
   the exhaustive-cache equivalent of the PDB scripts' missing-alignment
   filtering.
8. Optionally fail if any assigned representative does not have a matching
   `<rep_id>.npz` in `--alignment-arrays-dir`.
9. Add cluster IDs and cluster sizes with `add_cluster_data`. This requires
   `mmseqs`, so run inside the `of3-internal` conda environment.
10. Mark every chain and interface for validation metrics:
    `low_homology=True`, `metric_eligible=True`, and `use_metrics=True`.
11. Write the cache with `write_datacache_to_json`.

```bash
python examples/finetuning_2/dataset_generation/src/create_validation_cache.py \
  --metadata-cache "$DATASET_ROOT/preprocessed/metadata_val.json" \
  --preprocessed-dir "$DATASET_ROOT/preprocessed/structure_files" \
  --alignment-representatives-fasta "$DATASET_ROOT/alignments/alignment_representatives.fasta" \
  --alignment-arrays-dir "$DATASET_ROOT/alignment_arrays" \
  --output "$DATASET_ROOT/caches/val_cache.json" \
  --dataset-name "a71ev2a-small-test-val"
```
