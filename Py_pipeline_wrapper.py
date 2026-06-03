import os
import argparse
import subprocess

FINETUNE_ROOT='/scratch/s5h/omeir.s5h/projects/openbind/fragment_finetune_benchmark/openfold3-finetune'
OPENFOLD_REPO_ROOT='/lus/lfs1aip2/scratch/s5h/etowah.s5h/projects/openfold3-internal'

parser = argparse.ArgumentParser()

parser.add_argument('--fragalysis_dir', '-fd', help='Path to aligned_files/ directory in fragalysis')
parser.add_argument('--output_dir', '-od', help='Path to directory where preprocessing outputs will be stored')
parser.add_argument('--train_ids', '-tid', help='txt file with structure IDs for training set. One ID per line')
parser.add_argument('--val_ids', '-vid', help='txt file with structure IDs for validation set. One ID per line')
parser.add_argument('--dataset-name', '-n', help='Dataset name for the output weighted training dataset')

args = parser.parse_args()

def main():
    # Put everything in cif format
    merged_cifs_dir = os.path.abspath(f'{args.output_dir}/merged_cifs')
    cmd1 = f'python {FINETUNE_ROOT}/src/merge_pdb_sdfs_to_cif.py --input-dir {args.fragalysis_dir} --output-dir {merged_cifs_dir}'.split()
    subprocess.run(cmd1)

    # Create new CCD with new small molecules added
    cmd2 = f'python {OPENFOLD_REPO_ROOT}/scripts/data_preprocessing/preprocess_ccd_biotite.py {merged_cifs_dir}/reference_data/components_harmonized.cif {merged_cifs_dir}/reference_data/components_harmonized.bcif'.split()
    subprocess.run(cmd2)

    # Monomerize training data
    monomer_cifs_dir = f'{args.output_dir}/merged_cifs_monomer'
    cmd3 = f'python {FINETUNE_ROOT}/src/process_into_monomer.py --input-dir {merged_cifs_dir} --output-dir {monomer_cifs_dir} --log-path {monomer_cifs_dir}/process_into_monomer.log'.split()
    subprocess.run(cmd3)

    preprocessed_dir = f'{args.output_dir}/preprocessed'
    cmd4 = f'python {OPENFOLD_REPO_ROOT}/scripts/data_preprocessing/preprocess_pdb_of3.py --cif-dir {monomer_cifs_dir} --ccd-path {merged_cifs_dir}/reference_data/components_harmonized.cif --biotite-ccd-path {merged_cifs_dir}/reference_data/components_harmonized.bcif --out-dir {preprocessed_dir} --output-format npz --num-workers 0 --log-level INFO'.split()
    subprocess.run(cmd4)
    
    # Cluster structure sequences, and get representative sequences to use for subsequent MSA calculations
    alignments_dir = f'{args.output_dir}/alignments'
    cmd5 = f'python {FINETUNE_ROOT}/src/build_alignment_representatives.py --preprocessed-structure-dir {preprocessed_dir}/structure_files/ --output-fasta {alignments_dir}/alignment_representatives.fasta --report-json {alignments_dir}/alignment_representatives_report.json'.split()
    subprocess.run(cmd5)

    # Get ColabFold MSAs
    query_json_dir = f'{args.output_dir}/msa_queries'
    cmd6 = f'python {FINETUNE_ROOT}/src/generate_msas.py --representatives-fasta {alignments_dir}/alignment_representatives.fasta --output-dir {alignments_dir} --query-json-dir {query_json_dir}'.split()
    subprocess.run(cmd6)

    # Create alignment arrays
    aln_arrays = f'{args.output_dir}/alignment_arrays'
    os.makedirs(f'{aln_arrays}', exist_ok=True)
    cmd7 = f'python {OPENFOLD_REPO_ROOT}/scripts/data_preprocessing/preparse_alignments_of3.py --alignments_directory {alignments_dir}/main --alignment_array_directory {aln_arrays} --num_workers 1 --max_seq_counts'.split()
    #cmd7.append('{"cfdb_hits": 16384}')
    cmd7.append('{"colabfold_main": 16384}')
    subprocess.run(cmd7)

    # Materialize split metadat files
    cmd8 = f'python {FINETUNE_ROOT}/src/materialize_split_metadata.py --metadata-cache {preprocessed_dir}/metadata.json --train-ids {args.train_ids} --val-ids {args.val_ids} --train-output {preprocessed_dir}/metadata_train.json --val-output {preprocessed_dir}/metadata_val.json'.split()
    subprocess.run(cmd8)

    # Create the weighted training dataset cache
    cache_dir = f'{args.output_dir}/caches'
    os.makedirs(cache_dir, exist_ok=True)

    #cmd9 = f'python {OPENFOLD_REPO_ROOT}/scripts/data_preprocessing/create_pdb-weighted_training_dataset_cache.py --metadata-cache {preprocessed_dir}/metadata_train.json --preprocessed-dir {preprocessed_dir}/structure_files --alignment-representatives-fasta {alignments_dir}/alignment_representatives.fasta --output {cache_dir}/train_cache.json --dataset-name {args.dataset_name} --allow-missing-alignment --missing-alignment-log {cache_dir}/missing_alignments.json --log-level INFO'.split()

    cmd9 = f'python {OPENFOLD_REPO_ROOT}/scripts/data_preprocessing/create_pdb-weighted_training_dataset_cache.py --metadata-cache {preprocessed_dir}/metadata_train.json --preprocessed-dir {preprocessed_dir}/structure_files --alignment-representatives-fasta {alignments_dir}/alignment_representatives.fasta --output {cache_dir}/train_cache.json --dataset-name {args.dataset_name} --missing-alignment-log {cache_dir}/missing_alignments.json --log-level INFO'.split()
    subprocess.run(cmd9)

    # Create full validation cache
    cmd10 = f'python {FINETUNE_ROOT}/src/create_validation_cache.py --metadata-cache {preprocessed_dir}/metadata_val.json --preprocessed-dir {preprocessed_dir}/structure_files --alignment-representatives-fasta {alignments_dir}/alignment_representatives.fasta --alignment-arrays-dir {aln_arrays} --output {cache_dir}/val_cache.json --dataset-name {args.dataset_name}'.split()
    subprocess.run(cmd10)
    return

if __name__=='__main__':
    main()
