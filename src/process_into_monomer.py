#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from openfold3.core.data.io.structure.cif import (
    ParsedStructure,
    parse_mmcif,
    write_structure,
)
from utils import (
    _inject_required_cif_metadata,
    prepare_atom_array_for_of3_cif_write,
    sanitize_stem,
)


MAIN_LIGAND_EXCLUDED_RESNAMES = frozenset(
    {
        "DMS",
        "HOH",
        "WAT",
        "H2O",
        "DOD",
        "LI",
        "NA",
        "K",
        "RB",
        "CS",
        "MG",
        "CA",
        "SR",
        "BA",
        "ZN",
        "MN",
        "FE",
        "FE2",
        "FE3",
        "CO",
        "NI",
        "CU",
        "CU1",
        "CUA",
        "CUZ",
        "CD",
        "HG",
        "AL",
        "CL",
        "BR",
        "IOD",
        "F",
    }
)


@dataclass(frozen=True)
class HeteroResidue:
    chain_id: str
    res_id: int
    ins_code: str
    res_name: str
    atom_indices: np.ndarray

    @property
    def label(self) -> str:
        ins_code = self.ins_code if self.ins_code else "-"
        return f"{self.res_name}:{self.chain_id}:{self.res_id}:{ins_code}"


@dataclass(frozen=True)
class HeteroChain:
    chain_id: str
    residue_labels: tuple[str, ...]
    residue_count: int
    atom_count: int
    residue_names: frozenset[str]

    @property
    def summary(self) -> str:
        names = ",".join(sorted(self.residue_names))
        return (
            f"chain={self.chain_id} residues={self.residue_count} "
            f"atoms={self.atom_count} res_names={names}"
        )


@dataclass(frozen=True)
class ProcessResult:
    source_name: str
    selected_protein_chain: str
    main_ligand_chain: str
    kept_protein_chains: tuple[str, ...]
    removed_protein_chains: tuple[str, ...]
    kept_hetero_residues: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert merged CIFs into monomerized CIFs by keeping the single "
            "protein chain closest to the main multi-atom ligand and nearby "
            "hetero residues."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing merged CIFs from step 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write processed monomer CIFs.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=6.0,
        help="Contact radius in Angstroms for keeping extra hetero residues.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional plain-text log file.",
    )
    parser.add_argument(
        "--keep-only-main-ligand",
        action="store_true",
        help=(
            "Keep only the selected protein chain and the single main ligand chain. "
            "If unset, additional hetero residues within --radius of the protein chain "
            "are also kept."
        ),
    )
    return parser.parse_args()


def extract_cif_metadata(cif_path: Path) -> tuple[str, str, float | None]:
    release_date = "1970-01-01"
    experimental_method = "UNKNOWN"
    resolution = None

    cif_file = pdbx.CIFFile.read(cif_path)
    block = cif_file[list(cif_file.keys())[0]]

    if "pdbx_audit_revision_history" in block:
        revision_dates = block["pdbx_audit_revision_history"]["revision_date"].as_array()
        if len(revision_dates) > 0:
            release_date = str(revision_dates[-1])

    if "exptl" in block:
        methods = block["exptl"]["method"].as_array()
        if len(methods) > 0:
            experimental_method = str(methods[0])

    if "refine" in block and "ls_d_res_high" in block["refine"]:
        resolution_values = block["refine"]["ls_d_res_high"].as_array()
        if len(resolution_values) > 0:
            value = str(resolution_values[0]).strip()
            if value not in {"?", ".", ""}:
                resolution = float(value)

    return release_date, experimental_method, resolution


def _normalize_ins_code(value: object) -> str:
    text = str(value).strip()
    if text in {"?", "."}:
        return ""
    return text


def _iter_hetero_residues(atom_array: AtomArray) -> list[HeteroResidue]:
    residues: list[HeteroResidue] = []
    residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    has_ins_code = "ins_code" in atom_array.get_annotation_categories()

    for start, stop in zip(residue_starts[:-1], residue_starts[1:], strict=False):
        if not bool(atom_array.hetero[start]):
            continue
        residues.append(
            HeteroResidue(
                chain_id=str(atom_array.chain_id[start]),
                res_id=int(atom_array.res_id[start]),
                ins_code=_normalize_ins_code(atom_array.ins_code[start]) if has_ins_code else "",
                res_name=str(atom_array.res_name[start]).strip().upper(),
                atom_indices=np.arange(start, stop, dtype=np.int64),
            )
        )

    return residues


def _summarize_hetero_chains(residues: list[HeteroResidue]) -> dict[str, HeteroChain]:
    by_chain: dict[str, list[HeteroResidue]] = {}
    for residue in residues:
        by_chain.setdefault(residue.chain_id, []).append(residue)

    summaries: dict[str, HeteroChain] = {}
    for chain_id, chain_residues in sorted(by_chain.items()):
        summaries[chain_id] = HeteroChain(
            chain_id=chain_id,
            residue_labels=tuple(residue.label for residue in chain_residues),
            residue_count=len(chain_residues),
            atom_count=sum(len(residue.atom_indices) for residue in chain_residues),
            residue_names=frozenset(residue.res_name for residue in chain_residues),
        )

    return summaries


def _protein_chain_ids(atom_array: AtomArray) -> tuple[str, ...]:
    chain_ids = {
        str(atom_array.chain_id[start])
        for start in struc.get_residue_starts(atom_array)
        if not bool(atom_array.hetero[start])
    }
    return tuple(sorted(chain_ids))


def _main_ligand_chain_candidates(hetero_chains: dict[str, HeteroChain]) -> list[HeteroChain]:
    candidates: list[HeteroChain] = []
    for chain in hetero_chains.values():
        if chain.atom_count <= 1:
            continue
        if chain.residue_names.issubset(MAIN_LIGAND_EXCLUDED_RESNAMES):
            continue
        candidates.append(chain)
    return candidates


def _min_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    deltas = coords_a[:, None, :] - coords_b[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=-1))
    return float(np.min(distances))

def _find_closest_protein_chain(
    atom_array: AtomArray,
    protein_chains: tuple[str, ...],
    main_ligand_chain_id: str,
) -> tuple[str, float]:
    ligand_atoms = atom_array[(atom_array.chain_id == main_ligand_chain_id) & atom_array.hetero]
    if ligand_atoms.array_length() == 0:
        raise ValueError(f"No hetero atoms found for main ligand chain {main_ligand_chain_id}")

    distance_by_chain: list[tuple[str, float]] = []
    for chain_id in protein_chains:
        protein_atoms = atom_array[(atom_array.chain_id == chain_id) & (~atom_array.hetero)]
        if protein_atoms.array_length() == 0:
            continue
        distance_by_chain.append(
            (chain_id, _min_distance(ligand_atoms.coord, protein_atoms.coord))
        )

    if not distance_by_chain:
        raise ValueError("No protein atoms found while selecting closest chain")

    return min(distance_by_chain, key=lambda item: (item[1], item[0]))

def _count_contacts(coords_a: np.ndarray, coords_b: np.ndarray, cutoff: float) -> float:
    deltas = coords_a[:, None, :] - coords_b[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=-1))
    contact_count = np.sum(distances <= cutoff)

    return float(contact_count)

def _find_biggest_protein_interface(
    atom_array: AtomArray,
    protein_chains: tuple[str, ...],
    main_ligand_chain_id: str,
) -> tuple[str, float]:
    ligand_atoms = atom_array[(atom_array.chain_id == main_ligand_chain_id) & atom_array.hetero]

    if ligand_atoms.array_length() == 0:
        raise ValueError(f"No hetero atoms found for main ligand chain {main_ligand_chain_id}")

    contacts_by_chain: list[tuple[str, float]] = []
    for chain_id in protein_chains:
        protein_atoms = atom_array[(atom_array.chain_id == chain_id) & (~atom_array.hetero)]
        if protein_atoms.array_length() == 0:
            continue
        contacts_by_chain.append(
            (chain_id, _count_contacts(ligand_atoms.coord, protein_atoms.coord, 6.5))
        )
    
    if not contacts_by_chain:
        raise ValueError("No protein atoms found while selecting closest chain")

    return max(contacts_by_chain, key=lambda item: (item[1], item[0]))

def _residue_is_within_radius(
    residue: HeteroResidue,
    atom_array: AtomArray,
    protein_atoms: AtomArray,
    radius: float,
) -> bool:
    if protein_atoms.array_length() == 0:
        return False
    residue_atoms = atom_array[residue.atom_indices]
    cell_list = struc.CellList(protein_atoms, cell_size=radius)
    neighbor_idx = np.asarray(cell_list.get_atoms(residue_atoms.coord, radius=radius))
    return bool(np.any(neighbor_idx >= 0))


def _build_keep_mask(
    atom_array: AtomArray,
    selected_protein_chain: str,
    main_ligand_chain: str,
    radius: float,
    keep_only_main_ligand: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    keep_mask = np.zeros(atom_array.array_length(), dtype=bool)
    protein_mask = (~atom_array.hetero) & (atom_array.chain_id == selected_protein_chain)
    keep_mask |= protein_mask

    protein_atoms = atom_array[protein_mask]
    kept_hetero_residues: list[str] = []
    for residue in _iter_hetero_residues(atom_array):
        should_keep = False
        if residue.chain_id == main_ligand_chain:
            should_keep = True
        elif (not keep_only_main_ligand) and _residue_is_within_radius(
            residue=residue,
            atom_array=atom_array,
            protein_atoms=protein_atoms,
            radius=radius,
        ):
            should_keep = True

        if should_keep:
            keep_mask[residue.atom_indices] = True
            kept_hetero_residues.append(residue.label)

    return keep_mask, tuple(kept_hetero_residues)


def write_filtered_cif(atom_array: AtomArray, src_path: Path, dst_path: Path) -> None:
    filtered_array = prepare_atom_array_for_of3_cif_write(atom_array)
    release_date, experimental_method, resolution = extract_cif_metadata(src_path)
    write_structure(
        atom_array=filtered_array,
        output_path=dst_path,
        data_block=sanitize_stem(src_path.stem),
        include_bonds=True,
        make_ost_compatible=True,
    )
    _inject_required_cif_metadata(
        cif_path=dst_path,
        atom_array=filtered_array,
        release_date=release_date,
        experimental_method=experimental_method,
        resolution=resolution,
    )


def process_structure(
    src_path: Path,
    dst_path: Path,
    radius: float,
    keep_only_main_ligand: bool,
) -> ProcessResult:
    parsed = parse_mmcif(
        src_path,
        expand_bioassembly=False,
        include_bonds=True,
    )
    if not isinstance(parsed, ParsedStructure) or parsed.atom_array is None:
        raise ValueError(f"Failed to parse source CIF: {src_path}")

    atom_array = parsed.atom_array
    protein_chains = _protein_chain_ids(atom_array)
    if not protein_chains:
        raise ValueError("No protein chains found")

    hetero_residues = _iter_hetero_residues(atom_array)
    hetero_chains = _summarize_hetero_chains(hetero_residues)
    candidate_chains = _main_ligand_chain_candidates(hetero_chains)
    if len(candidate_chains) != 1:
        summaries = "; ".join(chain.summary for chain in candidate_chains) or "none"
        all_hetero = "; ".join(chain.summary for chain in hetero_chains.values()) or "none"
        raise ValueError(
            "Expected exactly one non-ion/non-DMS multi-atom ligand chain, "
            f"found {len(candidate_chains)} candidates. "
            f"candidates=[{summaries}] all_hetero=[{all_hetero}]"
        )

    main_ligand_chain = candidate_chains[0].chain_id
    #selected_protein_chain, _distance = _find_closest_protein_chain(
    #    atom_array=atom_array,
    #    protein_chains=protein_chains,
    #    main_ligand_chain_id=main_ligand_chain,
    #)
    selected_protein_chain, _distance = _find_biggest_protein_interface(
        atom_array=atom_array,
        protein_chains=protein_chains,
        main_ligand_chain_id=main_ligand_chain,
    )

    keep_mask, kept_hetero_residues = _build_keep_mask(
        atom_array=atom_array,
        selected_protein_chain=selected_protein_chain,
        main_ligand_chain=main_ligand_chain,
        radius=radius,
        keep_only_main_ligand=keep_only_main_ligand,
    )
    filtered_array = atom_array[keep_mask]
    write_filtered_cif(filtered_array, src_path=src_path, dst_path=dst_path)

    return ProcessResult(
        source_name=src_path.name,
        selected_protein_chain=selected_protein_chain,
        main_ligand_chain=main_ligand_chain,
        kept_protein_chains=(selected_protein_chain,),
        removed_protein_chains=tuple(
            chain_id for chain_id in protein_chains if chain_id != selected_protein_chain
        ),
        kept_hetero_residues=kept_hetero_residues,
    )


def write_log(
    log_path: Path,
    input_dir: Path,
    output_dir: Path,
    results: list[ProcessResult],
    failures: list[tuple[str, str]],
) -> None:
    lines = [
        f"input_dir={input_dir}",
        f"output_dir={output_dir}",
        f"processed_ok={len(results)}",
        f"processed_failed={len(failures)}",
        "",
    ]

    for result in results:
        lines.append(f"file={result.source_name}")
        lines.append(f"selected_protein_chain={result.selected_protein_chain}")
        lines.append(f"main_ligand_chain={result.main_ligand_chain}")
        lines.append(
            "removed_protein_chains="
            + (",".join(result.removed_protein_chains) if result.removed_protein_chains else "-")
        )
        lines.append(
            "kept_hetero_residues="
            + (",".join(result.kept_hetero_residues) if result.kept_hetero_residues else "-")
        )
        lines.append("")

    for source_name, error_text in failures:
        lines.append(f"file={source_name}")
        lines.append("status=failed")
        lines.append(f"error={error_text}")
        lines.append("")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    src_paths = sorted(input_dir.glob("*.cif"))
    if not src_paths:
        raise FileNotFoundError(f"No CIF files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ProcessResult] = []
    failures: list[tuple[str, str]] = []
    for src_path in src_paths:
        dst_path = output_dir / src_path.name
        try:
            result = process_structure(
                src_path=src_path,
                dst_path=dst_path,
                radius=args.radius,
                keep_only_main_ligand=args.keep_only_main_ligand,
            )
        except Exception as exc:
            failures.append((src_path.name, str(exc)))
            print(f"{src_path.name}: failed: {exc}", file=sys.stderr)
            continue

        results.append(result)
        removed_label = ",".join(result.removed_protein_chains) if result.removed_protein_chains else "-"
        print(
            f"{result.source_name}: kept_protein_chain={result.selected_protein_chain} "
            f"main_ligand_chain={result.main_ligand_chain} "
            f"removed_protein_chains={removed_label}"
        )

    if args.log_path is not None:
        write_log(
            log_path=args.log_path,
            input_dir=input_dir,
            output_dir=output_dir,
            results=results,
            failures=failures,
        )
        print(f"Log written to {args.log_path}")

    print(f"Processed {len(results)}/{len(src_paths)} CIFs into {output_dir}")
    if failures:
        print(f"{len(failures)} CIFs failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
