#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_ids(ids_path: Path) -> list[str]:
    seen = set()
    ids: list[str] = []

    with ids_path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line in seen:
                raise ValueError(
                    f"Duplicate structure id {line!r} in {ids_path} on line {line_number}"
                )
            seen.add(line)
            ids.append(line)

    if not ids:
        raise ValueError(f"No structure ids found in {ids_path}")

    return ids


def _build_subset(
    metadata: dict,
    selected_ids: list[str],
) -> dict:
    all_structure_data = metadata["structure_data"]
    missing_ids = [structure_id for structure_id in selected_ids if structure_id not in all_structure_data]
    if missing_ids:
        formatted = ", ".join(missing_ids)
        raise KeyError(f"Structure ids not found in metadata.json: {formatted}")

    subset_structure_data = {
        structure_id: all_structure_data[structure_id] for structure_id in selected_ids
    }
    all_reference_molecule_data = metadata.get("reference_molecule_data", {})

    return {
        "structure_data": subset_structure_data,
        # Keep the full reference-molecule table for the split. Downstream OF3
        # feature creation looks up polymer component conformers by residue name
        # (e.g. "SER"), not only by explicit chain-level reference_mol_id, so
        # filtering to ligand-only IDs breaks training/validation.
        "reference_molecule_data": all_reference_molecule_data,
    }


def _write_json(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Subset a preprocessing metadata.json into metadata_train.json and "
            "metadata_val.json using explicit structure-id lists."
        )
    )
    parser.add_argument(
        "--metadata-cache",
        type=Path,
        required=True,
        help="Input metadata.json from preprocess_pdb_of3.py.",
    )
    parser.add_argument(
        "--train-ids",
        type=Path,
        required=True,
        help="Text file with one training structure id per line.",
    )
    parser.add_argument(
        "--val-ids",
        type=Path,
        required=True,
        help="Text file with one validation structure id per line.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        required=True,
        help="Path to write metadata_train.json.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        required=True,
        help="Path to write metadata_val.json.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    with args.metadata_cache.open() as handle:
        metadata = json.load(handle)

    if "structure_data" not in metadata:
        raise KeyError(f"Missing structure_data in {args.metadata_cache}")
    if "reference_molecule_data" not in metadata:
        raise KeyError(f"Missing reference_molecule_data in {args.metadata_cache}")

    train_ids = _read_ids(args.train_ids)
    val_ids = _read_ids(args.val_ids)

    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(
            "Training and validation id lists overlap: " + ", ".join(overlap)
        )

    metadata_train = _build_subset(metadata, train_ids)
    metadata_val = _build_subset(metadata, val_ids)

    _write_json(args.train_output, metadata_train)
    _write_json(args.val_output, metadata_val)

    assigned_ids = set(train_ids) | set(val_ids)
    unassigned_ids = sorted(set(metadata["structure_data"]) - assigned_ids)
    print(
        f"Wrote {len(metadata_train['structure_data'])} training structures to "
        f"{args.train_output}"
    )
    print(
        f"Wrote {len(metadata_val['structure_data'])} validation structures to "
        f"{args.val_output}"
    )
    print(f"Left {len(unassigned_ids)} structures unassigned")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
