#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from openfold3.core.data.io.sequence.fasta import get_chain_id_to_seq_from_fasta
from openfold3.core.data.tools.colabfold_msa_server import query_colabfold_msa_server


logger = logging.getLogger("generate_msas")


def _read_first_a3m_query_sequence(a3m_path: Path) -> str | None:
    saw_header = False
    sequence_lines: list[str] = []
    with a3m_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if saw_header:
                    break
                saw_header = True
                continue
            if saw_header:
                sequence_lines.append(line)
    if not sequence_lines:
        return None
    return "".join(sequence_lines)


def _a3m_sequence_column_count(sequence: str) -> int:
    return sum(1 for char in sequence if not char.islower())


def _validate_a3m_query_length(
    rep_id: str,
    submitted_sequence: str,
    a3m_path: Path,
) -> tuple[bool, dict[str, Any]]:
    returned_sequence = _read_first_a3m_query_sequence(a3m_path)
    if returned_sequence is None:
        return False, {
            "representative_id": rep_id,
            "submitted_query_len": len(submitted_sequence),
            "returned_query_len": None,
            "returned_query_columns": None,
            "a3m_path": str(a3m_path),
            "reason": "Could not parse first query row from A3M",
        }

    returned_query_columns = _a3m_sequence_column_count(returned_sequence)
    result = {
        "representative_id": rep_id,
        "submitted_query_len": len(submitted_sequence),
        "returned_query_len": len(returned_sequence),
        "returned_query_columns": returned_query_columns,
        "a3m_path": str(a3m_path),
    }
    return returned_query_columns == len(submitted_sequence), result


def _expected_main_a3m(output_dir: Path, rep_id: str) -> Path:
    return output_dir / "main" / rep_id / "colabfold_main.a3m"


def _write_query_json(query_json_dir: Path, rep_id: str, sequence: str) -> Path:
    query_json_dir.mkdir(parents=True, exist_ok=True)
    query_json_path = query_json_dir / f"{rep_id}.json"
    payload = {
        "seeds": [42],
        "queries": {
            rep_id: {
                "chains": [
                    {
                        "molecule_type": "PROTEIN",
                        "chain_ids": ["A"],
                        "sequence": sequence,
                    }
                ]
            }
        },
    }
    query_json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return query_json_path


def _query_one_rep(
    rep_id: str,
    sequence: str,
    output_dir: Path,
    query_json_dir: Path,
    server_user_agent: str,
    server_url: str,
    force: bool,
) -> dict[str, Any]:
    out_a3m = _expected_main_a3m(output_dir, rep_id)
    if out_a3m.exists() and not force:
        is_valid, validation = _validate_a3m_query_length(rep_id, sequence, out_a3m)
        if is_valid:
            logger.info("Reusing valid MSA for %s: %s", rep_id, out_a3m)
            return {"representative_id": rep_id, "status": "reused", **validation}

        logger.warning(
            "Existing MSA failed length check for %s; regenerating. Details: %s",
            rep_id,
            validation,
        )
        out_a3m.unlink()

    raw_dir = output_dir / "raw" / "main" / rep_id
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    query_json_path = _write_query_json(query_json_dir, rep_id, sequence)
    logger.info("Querying ColabFold MSA server for %s", rep_id)
    a3m_lines = query_colabfold_msa_server(
        [sequence],
        prefix=raw_dir,
        use_templates=False,
        use_pairing=False,
        user_agent=server_user_agent,
        host_url=server_url,
    )
    if len(a3m_lines) != 1:
        raise RuntimeError(
            f"Expected exactly one A3M response for {rep_id}, got {len(a3m_lines)}"
        )

    out_a3m.parent.mkdir(parents=True, exist_ok=True)
    out_a3m.write_text(a3m_lines[0])
    is_valid, validation = _validate_a3m_query_length(rep_id, sequence, out_a3m)
    if not is_valid:
        out_a3m.unlink(missing_ok=True)
        raise RuntimeError(
            "Generated A3M query length mismatch for "
            f"{rep_id}: {json.dumps(validation, sort_keys=True)}"
        )

    logger.info("Wrote %s", out_a3m)
    return {
        "representative_id": rep_id,
        "status": "generated",
        "query_json_path": str(query_json_path),
        **validation,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ColabFold server MSAs for alignment representatives.",
    )
    parser.add_argument(
        "--representatives-fasta",
        type=Path,
        required=True,
        help="FASTA mapping representative IDs to query sequences.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Alignment output directory. Writes main/<rep_id>/colabfold_main.a3m.",
    )
    parser.add_argument(
        "--query-json-dir",
        type=Path,
        required=True,
        help="Directory where per-representative query JSON files are written.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path for a generation summary JSON.",
    )
    parser.add_argument(
        "--server-user-agent",
        default="openfold3-finetuning-v2",
        help="User-Agent sent to the ColabFold MSA server.",
    )
    parser.add_argument(
        "--server-url",
        default="https://api.colabfold.com",
        help="ColabFold MSA server URL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate MSAs even when an existing A3M passes the length check.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    rep_id_to_sequence = get_chain_id_to_seq_from_fasta(args.representatives_fasta)
    if not rep_id_to_sequence:
        raise ValueError(f"No representatives found in {args.representatives_fasta}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for rep_id, sequence in sorted(rep_id_to_sequence.items()):
        if "/" in rep_id:
            failures.append(
                {
                    "representative_id": rep_id,
                    "status": "failed",
                    "reason": "Representative IDs must not contain '/'",
                }
            )
            continue
        try:
            results.append(
                _query_one_rep(
                    rep_id=rep_id,
                    sequence=sequence,
                    output_dir=args.output_dir,
                    query_json_dir=args.query_json_dir,
                    server_user_agent=args.server_user_agent,
                    server_url=args.server_url,
                    force=args.force,
                )
            )
        except Exception as exc:
            logger.exception("MSA generation failed for %s", rep_id)
            failures.append(
                {
                    "representative_id": rep_id,
                    "status": "failed",
                    "reason": str(exc),
                }
            )

    report = {
        "representative_count": len(rep_id_to_sequence),
        "success_count": len(results),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n")

    print(
        "MSA generation complete: "
        f"{len(results)}/{len(rep_id_to_sequence)} representatives available "
        f"(failures={len(failures)})"
    )
    if failures:
        raise RuntimeError(
            f"MSA generation failed for {len(failures)} representatives; "
            f"see {args.report_json if args.report_json else 'log output'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
