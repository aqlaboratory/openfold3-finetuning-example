"""Helpers shared by the runners/ scripts."""
import os
import sys
import subprocess
from pathlib import Path

FINETUNE_ROOT = Path(__file__).resolve().parents[1]
OPENFOLD_REPO_ROOT_ENV = 'OPENFOLD_REPO_ROOT'


def resolve_openfold_repo(cli_value):
    """Path to the OpenFold3 repo, from --openfold_repo or $OPENFOLD_REPO_ROOT.

    There is deliberately no fallback location: guessing one only moves the
    failure to a confusing place deep in a subprocess.
    """
    value = cli_value or os.environ.get(OPENFOLD_REPO_ROOT_ENV)
    if not value:
        raise SystemExit(
            'No OpenFold3 repo given. Pass --openfold_repo /path/to/openfold3 '
            f'or set ${OPENFOLD_REPO_ROOT_ENV}.')
    repo = Path(value).expanduser().resolve()
    if not (repo / 'scripts' / 'data_preprocessing').is_dir():
        raise SystemExit(
            f'{repo} does not look like the OpenFold3 repo: no scripts/data_preprocessing/ in it.')
    return repo


def run(cmd, openfold_repo):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    # Put the interpreter's bin/ on PATH so tools installed in the same env (e.g. mmseqs) are found
    env = {**os.environ,
           'PATH': os.pathsep.join([os.path.dirname(sys.executable), os.environ.get('PATH', '')]),
           'PYTHONPATH': os.pathsep.join(filter(None, [str(openfold_repo), os.environ.get('PYTHONPATH')]))}
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def read_fasta(path):
    records, name = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            name = line[1:].split()[0]
            records[name] = ''
        elif name is not None:
            records[name] += line.strip()
    return records


def crop_a3m_record(seq, start, end):
    """Keep match columns [start, end) of an A3M row, plus the insertions between them."""
    out, col = [], 0
    for char in seq:
        if char.isupper() or char == '-':
            if start <= col < end:
                out.append(char)
            col += 1
        elif char.islower() and start < col < end:
            out.append(char)
    return ''.join(out)


def find_ungapped_offset(query, seq, max_mismatch_fraction):
    """Offset where seq aligns to query without gaps with the fewest mismatches (point mutants
    such as D39L, or a different construct N-terminus, still align this way)."""
    best = None
    for offset in range(len(query) - len(seq) + 1):
        mismatches = sum(a != b for a, b in zip(query[offset:offset + len(seq)], seq))
        if best is None or mismatches < best[1]:
            best = (offset, mismatches)
    if best is None or best[1] > max_mismatch_fraction * len(seq):
        return None
    return best


def write_cropped_msas(msa_path, representatives_fasta, alignments_main_dir, max_mismatch_fraction=0.1):
    """Write alignments/main/<rep_id>/colabfold_main.a3m for each representative by cropping
    a precomputed MSA to the representative's sequence. For point mutants the query row is
    replaced by the mutant sequence and the homologs are kept."""
    headers, seqs = [], []
    for line in Path(msa_path).read_text().splitlines():
        if line.startswith('>'):
            headers.append(line)
            seqs.append('')
        elif headers:
            seqs[-1] += line.strip()
    query = seqs[0]

    for rep_id, rep_seq in read_fasta(representatives_fasta).items():
        match = find_ungapped_offset(query, rep_seq, max_mismatch_fraction)
        if match is None:
            raise ValueError(f'{rep_id}: sequence does not align to the query of {msa_path} without gaps '
                             f'(<= {max_mismatch_fraction:.0%} mismatches); use ColabFold MSAs instead')
        start, mismatches = match
        end = start + len(rep_seq)
        lines = [headers[0], rep_seq]
        for header, seq in zip(headers[1:], seqs[1:]):
            cropped = crop_a3m_record(seq, start, end)
            if cropped.strip('-'):
                lines += [header, cropped]
        out_path = alignments_main_dir / rep_id / 'colabfold_main.a3m'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text('\n'.join(lines) + '\n')
        print(f'{rep_id}: wrote {len(lines) // 2} sequences ({mismatches} mismatches to query) to {out_path}')


def preparse_alignment_arrays(alignments_main_dir, alignment_array_dir, max_seq_counts, openfold_repo):
    """Same as openfold3's scripts/data_preprocessing/preparse_alignments_of3.py, which in the
    public repo rejects the 'colabfold_main' key in --max_seq_counts."""
    import numpy as np
    sys.path.insert(0, str(openfold_repo))
    from openfold3.core.data.io.sequence.msa import parse_msas_direct, standardize_filepaths

    alignment_array_dir.mkdir(parents=True, exist_ok=True)
    rep_dirs = sorted(p for p in alignments_main_dir.iterdir() if p.is_dir())
    for rep_dir in rep_dirs:
        msas = parse_msas_direct(file_list=standardize_filepaths(rep_dir), max_seq_counts=max_seq_counts)
        np.savez_compressed(alignment_array_dir / f'{rep_dir.name}.npz', **{k: v.to_dict() for k, v in msas.items()})
    print(f'Wrote {len(rep_dirs)} alignment arrays to {alignment_array_dir}')


def build_alignments(output_dir, openfold_repo, msa_a3m=None):
    """From <output_dir>/preprocessed, pick MSA representative sequences, get an MSA for each
    (by cropping msa_a3m if given, otherwise from the ColabFold server) and write alignment arrays."""
    python = sys.executable
    alignments_dir = output_dir / 'alignments'
    representatives_fasta = alignments_dir / 'alignment_representatives.fasta'

    # Cluster structure sequences, and get representative sequences to use for subsequent MSA calculations
    run([python, FINETUNE_ROOT / 'src/build_alignment_representatives.py',
         '--preprocessed-structure-dir', output_dir / 'preprocessed' / 'structure_files',
         '--output-fasta', representatives_fasta, '--report-json', alignments_dir / 'alignment_representatives_report.json'],
        openfold_repo)

    # Crop the precomputed MSA, or get ColabFold MSAs
    if msa_a3m is not None:
        write_cropped_msas(msa_a3m, representatives_fasta, alignments_dir / 'main')
    else:
        run([python, FINETUNE_ROOT / 'src/generate_msas.py', '--representatives-fasta', representatives_fasta,
             '--output-dir', alignments_dir, '--query-json-dir', output_dir / 'msa_queries'], openfold_repo)

    # Create alignment arrays
    preparse_alignment_arrays(alignments_dir / 'main', output_dir / 'alignment_arrays', {'colabfold_main': 16384}, openfold_repo)
