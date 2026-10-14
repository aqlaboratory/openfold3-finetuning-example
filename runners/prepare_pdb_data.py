"""Prepare a folder of mmCIF files downloaded from the PDB as OpenFold3 training data.

PDB entries already use standard CCD ligand codes, so unlike prepare_zenodo_data.py there is
no merging, synthetic CCD, or monomerization step: the CIFs go straight into OpenFold3's
preprocess_pdb_of3.py, followed by MSA representatives, MSAs and alignment arrays.
Train/val caches are created afterwards with runners/create_dataset_cache.py.
"""
import sys
import argparse
from pathlib import Path

from _common import FINETUNE_ROOT, build_alignments, resolve_openfold_repo, run

sys.path.insert(0, str(FINETUNE_ROOT / 'src'))
from utils import DEFAULT_CCD_URL, ensure_ccd_file

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', '-i', required=True, help='Folder of .cif files from the PDB')
parser.add_argument('--output_dir', '-o', required=True, help='Path to directory where preprocessing outputs will be stored')
parser.add_argument('--ccd_path', help='PDB Chemical Component Dictionary (components.cif). Downloaded to '
                                       '<output_dir>/reference_data/components.cif if not given')
parser.add_argument('--msa_a3m', help='Precomputed A3M MSA to crop to each structure\'s sequence (e.g. EV-A71_2A_cfdb_msa.a3m '
                                      'from the Zenodo release). If not given, MSAs are fetched from the ColabFold server')
parser.add_argument('--openfold_repo', help='Path to the OpenFold3 repo (required, or set $OPENFOLD_REPO_ROOT)')
parser.add_argument('--num_workers', type=int, default=1, help='Workers for OpenFold3 structure preprocessing')

args = parser.parse_args()


def main():
    args.openfold_repo = resolve_openfold_repo(args.openfold_repo)
    python = sys.executable
    openfold_scripts = Path(args.openfold_repo) / 'scripts' / 'data_preprocessing'
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    cif_paths = sorted(input_dir.glob('*.cif'))
    if not cif_paths:
        raise FileNotFoundError(f'No .cif files found in {input_dir}')

    # OpenFold3 names output folders after the file name but the files inside after the lowercase
    # PDB id, so symlink the inputs under lowercase names (RCSB downloads are e.g. 7H2T.cif)
    staged_dir = output_dir / 'input_cifs'
    staged_dir.mkdir(parents=True, exist_ok=True)
    for cif_path in cif_paths:
        dst = staged_dir / cif_path.name.lower()
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(cif_path)
    print(f'Staged {len(cif_paths)} CIFs into {staged_dir}')

    # Get the PDB's CCD and convert it for biotite
    ccd_cif = Path(args.ccd_path).resolve() if args.ccd_path else output_dir / 'reference_data' / 'components.cif'
    ensure_ccd_file(ccd_cif, ccd_download_url=DEFAULT_CCD_URL)
    ccd_bcif = output_dir / 'reference_data' / 'components.bcif'
    if not ccd_bcif.exists():
        ccd_bcif.parent.mkdir(parents=True, exist_ok=True)
        run([python, openfold_scripts / 'preprocess_ccd_biotite.py', ccd_cif, ccd_bcif], args.openfold_repo)

    # Preprocess into OpenFold3 structure files
    run([python, openfold_scripts / 'preprocess_pdb_of3.py', '--cif-dir', staged_dir, '--ccd-path', ccd_cif,
         '--biotite-ccd-path', ccd_bcif, '--out-dir', output_dir / 'preprocessed', '--output-format', 'npz', '--output-format', 'cif',
         '--num-workers', args.num_workers, '--log-level', 'INFO'], args.openfold_repo)

    # Representative sequences, MSAs and alignment arrays
    build_alignments(output_dir, args.openfold_repo, Path(args.msa_a3m).resolve() if args.msa_a3m else None)


if __name__ == '__main__':
    main()
