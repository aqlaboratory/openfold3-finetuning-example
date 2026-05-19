#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from openfold3.core.data.io.sequence.fasta import (
    consolidate_preprocessed_fastas,
    write_multichain_fasta,
)


def _build_representatives(id_to_sequence: dict[str, str]) -> dict[str, str]:
    sequence_to_ids: dict[str, list[str]] = defaultdict(list)
    for sequence_id, sequence in sorted(id_to_sequence.items()):
        sequence_to_ids[sequence].append(sequence_id)

    return {
        sequence_ids[0]: sequence
        for sequence, sequence_ids in sorted(
            sequence_to_ids.items(), key=lambda item: item[1][0]
        )
    }


def _write_report(
    report_path: Path,
    id_to_sequence: dict[str, str],
    representative_to_sequence: dict[str, str],
) -> None:
    sequence_to_ids: dict[str, list[str]] = defaultdict(list)
    for sequence_id, sequence in sorted(id_to_sequence.items()):
        sequence_to_ids[sequence].append(sequence_id)

    report = {
        "chain_count": len(id_to_sequence),
        "representative_count": len(representative_to_sequence),
        "representatives": [
            {
                "representative_id": sequence_ids[0],
                "sequence_length": len(sequence),
                "member_ids": sequence_ids,
            }
            for sequence, sequence_ids in sorted(
                sequence_to_ids.items(), key=lambda item: item[1][0]
            )
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deduplicated alignment representatives FASTA from OpenFold3 "
            "preprocessed structure FASTA files."
        )
    )
    parser.add_argument(
        "--preprocessed-structure-dir",
        type=Path,
        required=True,
        help="Directory containing one preprocessed structure subdirectory per sample.",
    )
    parser.add_argument(
        "--output-fasta",
        type=Path,
        required=True,
        help="Path to write alignment_representatives.fasta.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional JSON report listing representative members.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    id_to_sequence = consolidate_preprocessed_fastas(args.preprocessed_structure_dir)
    if not id_to_sequence:
        raise ValueError(
            f"No FASTA sequences found under {args.preprocessed_structure_dir}"
        )

    representative_to_sequence = _build_representatives(id_to_sequence)

    args.output_fasta.parent.mkdir(parents=True, exist_ok=True)
    write_multichain_fasta(args.output_fasta, representative_to_sequence, sort=True)

    if args.report_json is not None:
        _write_report(args.report_json, id_to_sequence, representative_to_sequence)

    print(
        "Wrote "
        f"{len(representative_to_sequence)} representatives from "
        f"{len(id_to_sequence)} chains to {args.output_fasta}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
