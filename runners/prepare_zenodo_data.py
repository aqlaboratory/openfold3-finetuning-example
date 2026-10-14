"""Prepare the OpenBind EV-A71 2A Zenodo release (https://zenodo.org/records/22142262)
as OpenFold3 training data.

The Zenodo release nests each binding event as
    structures/<compound_group>/<complex_id>/<complex_id>_complex_ref.pdb
                                            <complex_id>_ligand_ref.sdf
so we first stage it into the Fragalysis-style layout that merge_pdb_sdfs_to_cif.py
expects (<complex_id>/<complex_id>.pdb + <complex_id>_ligand.sdf), then run preprocessing
up to (and including) alignment arrays.
Train/val caches are created afterwards with runners/create_dataset_cache.py.
"""
import sys
import argparse
from pathlib import Path

from _common import FINETUNE_ROOT, build_alignments, resolve_openfold_repo, run

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', '-i', required=True, help='Path to the unzipped Zenodo release (the OpenBind_EV-A71_2A/ directory)')
parser.add_argument('--output_dir', '-o', required=True, help='Path to directory where preprocessing outputs will be stored')
parser.add_argument('--openfold_repo', help='Path to the OpenFold3 repo (required, or set $OPENFOLD_REPO_ROOT)')
parser.add_argument('--msa', choices=['provided', 'colabfold'], default='provided',
                    help="'provided': crop the MSA shipped with the release to each structure's sequence. "
                         "'colabfold': query the ColabFold MSA server instead")
parser.add_argument('--num_workers', type=int, default=1, help='Workers for OpenFold3 structure preprocessing')

args = parser.parse_args()


def stage_structures(input_dir, staged_dir):
    """Symlink the Zenodo structures into <complex_id>/<complex_id>.pdb + <complex_id>_ligand.sdf."""
    staged_dir.mkdir(parents=True, exist_ok=True)
    complex_dirs = sorted(p for p in (input_dir / 'structures').glob('*/*') if p.is_dir())
    if not complex_dirs:
        raise FileNotFoundError(f'No structures found under {input_dir / "structures"}')

    n_staged = 0
    for complex_dir in complex_dirs:
        complex_id = complex_dir.name
        pdb_path = complex_dir / f'{complex_id}_complex_ref.pdb'
        sdf_path = complex_dir / f'{complex_id}_ligand_ref.sdf'
        if not (pdb_path.exists() and sdf_path.exists()):
            print(f'Skipping {complex_dir}: missing {pdb_path.name} or {sdf_path.name}')
            continue
        out_dir = staged_dir / complex_id
        out_dir.mkdir(exist_ok=True)
        for src, dst in [(pdb_path, out_dir / f'{complex_id}.pdb'), (sdf_path, out_dir / f'{complex_id}_ligand.sdf')]:
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        n_staged += 1
    print(f'Staged {n_staged} complexes into {staged_dir}')


def main():
    args.openfold_repo = resolve_openfold_repo(args.openfold_repo)
    python = sys.executable
    openfold_scripts = Path(args.openfold_repo) / 'scripts' / 'data_preprocessing'
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Put the Zenodo structures in the layout merge_pdb_sdfs_to_cif.py expects
    staged_dir = output_dir / 'input_structures'
    stage_structures(input_dir, staged_dir)

    # Put everything in cif format
    merged_cifs_dir = output_dir / 'merged_cifs'
    run([python, FINETUNE_ROOT / 'src/merge_pdb_sdfs_to_cif.py', '--input-dir', staged_dir, '--output-dir', merged_cifs_dir], args.openfold_repo)

    # Create new CCD with new small molecules added
    ccd_cif = merged_cifs_dir / 'reference_data/components_harmonized.cif'
    ccd_bcif = merged_cifs_dir / 'reference_data/components_harmonized.bcif'
    run([python, openfold_scripts / 'preprocess_ccd_biotite.py', ccd_cif, ccd_bcif], args.openfold_repo)

    # Monomerize training data
    monomer_cifs_dir = output_dir / 'merged_cifs_monomer'
    run([python, FINETUNE_ROOT / 'src/process_into_monomer.py', '--input-dir', merged_cifs_dir, '--output-dir', monomer_cifs_dir,
         '--log-path', monomer_cifs_dir / 'process_into_monomer.log'], args.openfold_repo)

    # Preprocess into OpenFold3 structure files
    preprocessed_dir = output_dir / 'preprocessed'
    run([python, openfold_scripts / 'preprocess_pdb_of3.py', '--cif-dir', monomer_cifs_dir, '--ccd-path', ccd_cif,
         '--biotite-ccd-path', ccd_bcif, '--out-dir', preprocessed_dir, '--output-format', 'npz', '--output-format', 'cif',
         '--num-workers', args.num_workers, '--log-level', 'INFO'], args.openfold_repo)

    # Representative sequences, MSAs and alignment arrays
    msa_a3m = input_dir / 'EV-A71_2A_cfdb_msa.a3m' if args.msa == 'provided' else None
    build_alignments(output_dir, args.openfold_repo, msa_a3m)


if __name__ == '__main__':
    main()
