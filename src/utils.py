from __future__ import annotations

import datetime as dt
import gzip
import logging
import re
import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np

DEFAULT_CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"


def sanitize_stem(stem: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._").lower()
    return out if out else "structure"


def ensure_ccd_file(
    ccd_path: Path,
    ccd_download_url: str = DEFAULT_CCD_URL,
    logger: logging.Logger | None = None,
) -> None:
    if ccd_path.exists():
        return

    ccd_path.parent.mkdir(parents=True, exist_ok=True)
    if logger is not None:
        logger.info("Downloading CCD to %s", ccd_path)

    with urlopen(ccd_download_url) as response:
        with gzip.GzipFile(fileobj=response) as gz_file:
            with open(ccd_path, "wb") as out_file:
                shutil.copyfileobj(gz_file, out_file)

    if logger is not None:
        logger.info("CCD download complete")


def extract_pdb_metadata(pdb_path: Path) -> tuple[str, str, float | None]:
    """Extract minimal metadata needed by OF3 preprocessing from PDB text."""
    release_date = "1970-01-01"
    experimental_method = "UNKNOWN"
    resolution: float | None = None

    for line in pdb_path.read_text(errors="ignore").splitlines():
        if line.startswith("HEADER"):
            maybe_date = line[50:59].strip()
            if maybe_date:
                try:
                    release_date = dt.datetime.strptime(maybe_date, "%d-%b-%y").strftime(
                        "%Y-%m-%d"
                    )
                except ValueError:
                    pass
        elif line.startswith("EXPDTA"):
            method = line[10:].strip()
            if method:
                experimental_method = method.split(";")[0].strip().upper()
        elif line.startswith("REMARK   2 RESOLUTION."):
            match = re.search(r"RESOLUTION\.\s+([0-9.]+)\s+ANGSTROMS", line)
            if match:
                try:
                    resolution = float(match.group(1))
                except ValueError:
                    resolution = None
        elif line.startswith("REMARK   3") and "RESOLUTION RANGE HIGH" in line:
            match = re.search(r":\s*([0-9.]+)\s*$", line)
            if match:
                try:
                    resolution = float(match.group(1))
                except ValueError:
                    resolution = None

    return release_date, experimental_method, resolution


def _inject_required_cif_metadata(
    cif_path: Path,
    atom_array,
    release_date: str,
    experimental_method: str,
    resolution: float | None,
    forced_experimental_method: str | None = None,
) -> None:
    """Add mmCIF categories required by OF3 parser if missing."""
    from biotite.structure.io import pdbx
    from biotite.structure.io.pdbx import CIFCategory
    from openfold3.core.data.resources.residues import (
        DNA_RESTYPE_3TO1,
        PROTEIN_RESTYPE_3TO1,
        RNA_RESTYPE_3TO1,
        MoleculeType,
    )

    cif_file = pdbx.CIFFile.read(cif_path)
    (block_name,) = cif_file.keys()
    block = cif_file[block_name]

    if "pdbx_audit_revision_history" not in block:
        block["pdbx_audit_revision_history"] = CIFCategory(
            {
                "ordinal": np.array(["1"], dtype=str),
                "data_content_type": np.array(["Structure model"], dtype=str),
                "major_revision": np.array(["1"], dtype=str),
                "minor_revision": np.array(["0"], dtype=str),
                "revision_date": np.array([release_date], dtype=str),
            }
        )

    method_to_write = (
        forced_experimental_method.strip().upper()
        if forced_experimental_method is not None
        else experimental_method
    )

    if "exptl" not in block or forced_experimental_method is not None:
        block["exptl"] = CIFCategory(
            {
                "entry_id": np.array(["unknown"], dtype=str),
                "method": np.array([method_to_write], dtype=str),
            }
        )

    if resolution is not None and "refine" not in block:
        block["refine"] = CIFCategory(
            {
                "ls_d_res_high": np.array([f"{resolution:.3f}"], dtype=str),
            }
        )

    if "entity_poly_seq" not in block or "entity_poly" not in block:
        import biotite.structure as struc

        entity_poly_seq_rows: list[tuple[str, str, str]] = []
        entity_poly_rows: list[tuple[str, str]] = []
        seen_entities: set[str] = set()

        for chain in struc.chain_iter(atom_array):
            entity_id = str(int(chain.entity_id[0]))
            molecule_type = int(np.bincount(chain.molecule_type_id).argmax())
            if molecule_type not in (
                int(MoleculeType.PROTEIN),
                int(MoleculeType.RNA),
                int(MoleculeType.DNA),
            ):
                continue

            monomers: list[str] = []
            one_letter_codes: list[str] = []

            for i, residue in enumerate(struc.residue_iter(chain), start=1):
                mon_id = str(residue.res_name[0])
                monomers.append(mon_id)
                entity_poly_seq_rows.append((entity_id, str(i), mon_id))

                if molecule_type == int(MoleculeType.PROTEIN):
                    one_letter_codes.append(PROTEIN_RESTYPE_3TO1.get(mon_id, "X"))
                elif molecule_type == int(MoleculeType.RNA):
                    one_letter_codes.append(RNA_RESTYPE_3TO1.get(mon_id, "N"))
                else:
                    one_letter_codes.append(DNA_RESTYPE_3TO1.get(mon_id, "N"))

            if monomers and entity_id not in seen_entities:
                entity_poly_rows.append((entity_id, "".join(one_letter_codes)))
                seen_entities.add(entity_id)

        if entity_poly_seq_rows and "entity_poly_seq" not in block:
            block["entity_poly_seq"] = CIFCategory(
                {
                    "entity_id": np.array([row[0] for row in entity_poly_seq_rows], dtype=str),
                    "num": np.array([row[1] for row in entity_poly_seq_rows], dtype=str),
                    "mon_id": np.array([row[2] for row in entity_poly_seq_rows], dtype=str),
                }
            )

        if entity_poly_rows and "entity_poly" not in block:
            block["entity_poly"] = CIFCategory(
                {
                    "entity_id": np.array([row[0] for row in entity_poly_rows], dtype=str),
                    "pdbx_seq_one_letter_code_can": np.array(
                        [row[1] for row in entity_poly_rows], dtype=str
                    ),
                }
            )

    cif_file.write(cif_path)


def prepare_atom_array_for_of3_cif_write(atom_array):
    """Reassign chain/entity annotations expected by the OF3 CIF writer."""
    import biotite.structure as struc

    from openfold3.core.data.resources.residues import MoleculeType

    atom_array = atom_array.copy()

    # Reassign chain IDs to avoid mixed polymer+ligand chains from auth_asym IDs.
    # This mirrors mmCIF-style separation where ligand components have distinct labels.
    n_atoms = len(atom_array)
    reassigned_chain_ids = np.array([""] * n_atoms, dtype=object)
    molecule_type_ids = np.zeros(n_atoms, dtype=int)

    next_chain = 1
    next_entity = 1
    chain_label_to_entity_id: dict[str, int] = {}

    polymer_chain_label_by_auth_chain: dict[str, str] = {}
    ligand_component_label_by_key: dict[tuple[str, int, str, str], str] = {}
    next_polymer_res_id_by_auth_chain: dict[str, int] = {}

    residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(residue_starts[:-1], residue_starts[1:], strict=False):
        auth_chain = str(atom_array.chain_id[start])
        res_id = int(atom_array.res_id[start])
        ins_code = (
            str(atom_array.ins_code[start])
            if "ins_code" in atom_array.get_annotation_categories()
            else "?"
        )
        res_name = str(atom_array.res_name[start])
        hetero = bool(atom_array.hetero[start])

        if not hetero:
            if auth_chain not in polymer_chain_label_by_auth_chain:
                label = str(next_chain)
                next_chain += 1
                polymer_chain_label_by_auth_chain[auth_chain] = label
                chain_label_to_entity_id[label] = next_entity
                next_entity += 1
            label = polymer_chain_label_by_auth_chain[auth_chain]
            moltype = int(MoleculeType.PROTEIN)
            dense_res_id = next_polymer_res_id_by_auth_chain.get(auth_chain, 1)
            atom_array.res_id[start:stop] = dense_res_id
            if "ins_code" in atom_array.get_annotation_categories():
                atom_array.ins_code[start:stop] = ""
            next_polymer_res_id_by_auth_chain[auth_chain] = dense_res_id + 1
        else:
            key = (auth_chain, res_id, ins_code, res_name)
            if key not in ligand_component_label_by_key:
                label = str(next_chain)
                next_chain += 1
                ligand_component_label_by_key[key] = label
                chain_label_to_entity_id[label] = next_entity
                next_entity += 1
            label = ligand_component_label_by_key[key]
            moltype = int(MoleculeType.LIGAND)

        reassigned_chain_ids[start:stop] = label
        molecule_type_ids[start:stop] = moltype

    atom_array.chain_id = reassigned_chain_ids.astype(str)
    atom_array.set_annotation("molecule_type_id", molecule_type_ids)

    entity_ids = np.array(
        [chain_label_to_entity_id[str(c)] for c in atom_array.chain_id], dtype=int
    )
    atom_array.set_annotation("entity_id", entity_ids)

    if "charge" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("charge", np.zeros(len(atom_array), dtype=int))
    else:
        atom_array.set_annotation("charge", np.asarray(atom_array.charge, dtype=int))
    atom_array.set_annotation("occupancy", np.ones(len(atom_array), dtype=float))

    return atom_array
