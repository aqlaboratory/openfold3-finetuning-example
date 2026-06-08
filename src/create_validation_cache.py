#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from openfold3.core.data.io.dataset_cache import write_datacache_to_json
from openfold3.core.data.io.sequence.fasta import (
    consolidate_preprocessed_fastas,
    get_chain_id_to_seq_from_fasta,
)
from openfold3.core.data.primitives.caches.clustering import add_cluster_data
from openfold3.core.data.primitives.caches.filtering import (
    add_chain_representatives,
    build_provisional_clustered_val_dataset_cache,
)
from openfold3.core.data.primitives.caches.format import PreprocessingDataCache
from openfold3.core.data.resources.residues import MoleculeType


ALIGNMENT_REQUIRED_MOLECULE_TYPES = {MoleculeType.PROTEIN, MoleculeType.RNA}


def _validate_metadata_cache(
    metadata_cache: PreprocessingDataCache,
    preprocessed_dir: Path,
) -> None:
    invalid_structures = []
    missing_files = []
    nonfinite_resolutions = []

    for structure_id, structure_data in sorted(metadata_cache.structure_data.items()):
        if structure_data.status != "success":
            invalid_structures.append((structure_id, structure_data.status))
            continue

        if (
            structure_data.resolution is not None
            and not math.isfinite(structure_data.resolution)
        ):
            nonfinite_resolutions.append(structure_id)

        structure_dir = preprocessed_dir / structure_id
        expected_paths = [
            structure_dir / f"{structure_id}.fasta",
            structure_dir / f"{structure_id}.npz",
        ]
        missing_files.extend(path for path in expected_paths if not path.exists())

    errors = []
    if invalid_structures:
        formatted = ", ".join(
            f"{structure_id}={status}" for structure_id, status in invalid_structures
        )
        errors.append(f"Metadata cache contains non-success structures: {formatted}")
    #if nonfinite_resolutions:
    #    errors.append(
    #        "Metadata cache contains non-finite resolutions: "
    #        + ", ".join(nonfinite_resolutions)
    #    )
    if missing_files:
        formatted = "\n  ".join(str(path) for path in missing_files)
        errors.append(f"Missing preprocessed FASTA/NPZ files:\n  {formatted}")

    if errors:
        raise ValueError("\n".join(errors))


def _validate_alignment_representatives(dataset_cache) -> set[str]:
    missing_representatives = []
    assigned_representatives = set()

    for structure_id, structure_data in sorted(dataset_cache.structure_data.items()):
        for chain_id, chain_data in sorted(structure_data.chains.items()):
            if chain_data.molecule_type not in ALIGNMENT_REQUIRED_MOLECULE_TYPES:
                continue

            representative_id = chain_data.alignment_representative_id
            if representative_id is None:
                missing_representatives.append(f"{structure_id}_{chain_id}")
            else:
                assigned_representatives.add(representative_id)

    if missing_representatives:
        formatted = "\n  ".join(missing_representatives)
        raise ValueError(
            "Protein/RNA chains missing alignment representatives:\n"
            f"  {formatted}"
        )

    return assigned_representatives


def _validate_alignment_arrays(
    alignment_arrays_dir: Path,
    assigned_representatives: set[str],
) -> None:
    missing_arrays = [
        alignment_arrays_dir / f"{representative_id}.npz"
        for representative_id in sorted(assigned_representatives)
        if not (alignment_arrays_dir / f"{representative_id}.npz").exists()
    ]
    if missing_arrays:
        formatted = "\n  ".join(str(path) for path in missing_arrays)
        raise ValueError(f"Missing preparsed alignment arrays:\n  {formatted}")


def _mark_all_metrics(dataset_cache) -> None:
    for structure_data in dataset_cache.structure_data.values():
        for chain_data in structure_data.chains.values():
            chain_data.low_homology = True
            chain_data.metric_eligible = True
            chain_data.use_metrics = True

        for interface_data in structure_data.interfaces.values():
            interface_data.low_homology = True
            interface_data.metric_eligible = True
            interface_data.use_metrics = True


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an exhaustive OpenFold3 validation cache from a preprocessing "
            "metadata cache without AF3/PDB validation-set filtering."
        )
    )
    parser.add_argument(
        "--metadata-cache",
        type=Path,
        required=True,
        help="Validation metadata.json written by preprocess_pdb_of3.py.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        required=True,
        help="Validation structure_files directory.",
    )
    parser.add_argument(
        "--alignment-representatives-fasta",
        type=Path,
        required=True,
        help="FASTA containing deduplicated MSA representative sequences.",
    )
    parser.add_argument(
        "--alignment-arrays-dir",
        type=Path,
        default=None,
        help="Optional directory containing <representative_id>.npz alignment arrays.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write val_cache.json.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Dataset name stored in the validation cache.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    metadata_cache = PreprocessingDataCache.from_json(args.metadata_cache)
    _validate_metadata_cache(metadata_cache, args.preprocessed_dir)

    id_to_sequence = consolidate_preprocessed_fastas(args.preprocessed_dir)
    if not id_to_sequence:
        raise ValueError(f"No FASTA sequences found under {args.preprocessed_dir}")

    dataset_cache = build_provisional_clustered_val_dataset_cache(
        preprocessing_cache=metadata_cache,
        dataset_name=args.dataset_name,
    )

    representative_to_sequence = get_chain_id_to_seq_from_fasta(
        args.alignment_representatives_fasta
    )
    add_chain_representatives(
        structure_cache=dataset_cache.structure_data,
        query_chain_to_seq=id_to_sequence,
        repr_chain_to_seq=representative_to_sequence,
    )
    assigned_representatives = _validate_alignment_representatives(dataset_cache)

    if args.alignment_arrays_dir is not None:
        _validate_alignment_arrays(args.alignment_arrays_dir, assigned_representatives)

    add_cluster_data(dataset_cache=dataset_cache, id_to_sequence=id_to_sequence)
    _mark_all_metrics(dataset_cache)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_datacache_to_json(dataset_cache, args.output)

    print(
        "Wrote full validation cache with "
        f"{len(dataset_cache.structure_data)} structures and "
        f"{len(assigned_representatives)} alignment representatives to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
