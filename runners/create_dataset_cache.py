"""Create an OpenFold3 training or validation dataset cache from a list of structure ids.

Run after runners/prepare_zenodo_data.py (or any pipeline producing the same --output_dir layout):
    python runners/create_dataset_cache.py train --ids train_ids.txt --dataset_dir <dataset_dir>
    python runners/create_dataset_cache.py validation --ids val_ids.txt --dataset_dir <dataset_dir>

Writes <dataset_dir>/caches/train_cache.json or val_cache.json by default.
"""
import sys
import json
import argparse
from pathlib import Path

from _common import FINETUNE_ROOT, resolve_openfold_repo, run

sys.path.insert(0, str(FINETUNE_ROOT / 'src'))
from materialize_split_metadata import _build_subset, _read_ids, _write_json

DEFAULT_OUTPUT_NAMES = {'train': 'train_cache.json', 'validation': 'val_cache.json'}

parser = argparse.ArgumentParser()

parser.add_argument('split', choices=['train', 'validation'], help='Which kind of cache to create')
parser.add_argument('--ids', required=True, help='txt file with structure IDs, one per line (e.g. a71ev2a-x0188a)')
parser.add_argument('--dataset_dir', required=True, help='Output directory of runners/prepare_zenodo_data.py')
parser.add_argument('--output', help='Path of the cache to write (default: <dataset_dir>/caches/train_cache.json or val_cache.json)')
parser.add_argument('--dataset_name', help='Dataset name stored in the cache (default: <dataset_dir name>-train or -val)')
parser.add_argument('--openfold_repo', help='Path to the OpenFold3 repo (required, or set $OPENFOLD_REPO_ROOT)')

args = parser.parse_args()


def main():
    args.openfold_repo = resolve_openfold_repo(args.openfold_repo)
    python = sys.executable
    dataset_dir = Path(args.dataset_dir).resolve()
    preprocessed_dir = dataset_dir / 'preprocessed'
    representatives_fasta = dataset_dir / 'alignments' / 'alignment_representatives.fasta'
    output = Path(args.output).resolve() if args.output else dataset_dir / 'caches' / DEFAULT_OUTPUT_NAMES[args.split]
    dataset_name = args.dataset_name or f"{dataset_dir.name}-{'train' if args.split == 'train' else 'val'}"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Subset the preprocessing metadata to the requested structure ids
    with open(preprocessed_dir / 'metadata.json') as handle:
        metadata = json.load(handle)
    ids = _read_ids(Path(args.ids))
    split_metadata = output.parent / f'{output.stem}_metadata.json'
    _write_json(split_metadata, _build_subset(metadata, ids))
    print(f'Wrote metadata for {len(ids)} structures to {split_metadata}')

    if args.split == 'train':
        # Create the weighted training dataset cache
        run([python, Path(args.openfold_repo) / 'scripts/data_preprocessing/create_pdb-weighted_training_dataset_cache.py',
             '--metadata-cache', split_metadata, '--preprocessed-dir', preprocessed_dir / 'structure_files',
             '--alignment-representatives-fasta', representatives_fasta, '--output', output, '--dataset-name', dataset_name,
             '--missing-alignment-log', output.parent / f'{output.stem}_missing_alignments.json', '--log-level', 'INFO'], args.openfold_repo)
    else:
        # Create full validation cache
        run([python, FINETUNE_ROOT / 'src/create_validation_cache.py', '--metadata-cache', split_metadata,
             '--preprocessed-dir', preprocessed_dir / 'structure_files', '--alignment-representatives-fasta', representatives_fasta,
             '--alignment-arrays-dir', dataset_dir / 'alignment_arrays', '--output', output, '--dataset-name', dataset_name], args.openfold_repo)

    with open(output) as handle:
        n_structures = len(json.load(handle)['structure_data'])
    print(f'Wrote {args.split} cache with {n_structures} structures to {output}')


if __name__ == '__main__':
    main()
