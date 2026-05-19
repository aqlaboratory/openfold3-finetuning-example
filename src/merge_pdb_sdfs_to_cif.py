#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, BondType
from biotite.structure.io import mol, pdb, pdbx
from biotite.structure.io.pdbx import CIFBlock, CIFCategory
from tqdm import tqdm

from utils import (
    DEFAULT_CCD_URL,
    _inject_required_cif_metadata,
    ensure_ccd_file,
    extract_pdb_metadata,
    prepare_atom_array_for_of3_cif_write,
    sanitize_stem,
)

GENERIC_LIGAND_REFERENCE_IDS = frozenset({"LIG", "UNL"})

@dataclass(frozen=True)
class ResidueKey:
    chain_id: str
    res_id: int
    ins_code: str
    res_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "res_id": self.res_id,
            "ins_code": self.ins_code,
            "res_name": self.res_name,
        }

@dataclass(frozen=True)
class SDFLigand:
    path: Path
    atom_array: AtomArray
    source_hint: ResidueKey | None


@dataclass(frozen=True)
class ComponentAssignment:
    component_id: str
    component_kind: Literal["base", "synthetic"]
    source_res_name: str


@dataclass(frozen=True)
class SyntheticComponent:
    atom_names: tuple[str, ...]
    atom_elements: tuple[str, ...]
    atom_charges: tuple[int, ...]
    atom_coords: tuple[tuple[float, float, float], ...]
    bonds: tuple[tuple[str, str, str, str], ...]


@dataclass(frozen=True)
class MergeResult:
    summary: dict[str, Any]
    synthetic_components: dict[str, SyntheticComponent]


def _normalize_element(value: Any) -> str:
    return str(value).strip().upper()


def _normalize_ins_code(value: Any) -> str:
    text = str(value).strip()
    if text in {"?", "."}:
        return ""
    return text


def _residue_key_from_array(atom_array: AtomArray) -> ResidueKey:
    if atom_array.array_length() == 0:
        raise ValueError("Residue array is empty")
    return ResidueKey(
        chain_id=str(atom_array.chain_id[0]).strip(),
        res_id=int(atom_array.res_id[0]),
        ins_code=_normalize_ins_code(atom_array.ins_code[0]),
        res_name=str(atom_array.res_name[0]).strip(),
    )


def _iter_residues(atom_array: AtomArray):
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(starts[:-1], starts[1:], strict=False):
        yield atom_array[start:stop]


def _parse_sdf_source_hint(sdf_path: Path) -> ResidueKey | None:
    match = re.search(
        r"(?P<res_name>[A-Za-z0-9]+)-(?P<res_id>-?\d+)-(?P<chain_id>[A-Za-z0-9]+)-lig$",
        sdf_path.stem,
    )
    if match is None:
        return None
    return ResidueKey(
        chain_id=match.group("chain_id"),
        res_id=int(match.group("res_id")),
        ins_code="",
        res_name=match.group("res_name").upper(),
    )


def _make_sdf_ligand(sdf_path: Path) -> SDFLigand:
    sdf_file = mol.SDFile.read(sdf_path)
    atom_array = mol.get_structure(sdf_file)
    return SDFLigand(
        path=sdf_path,
        atom_array=atom_array.copy(),
        source_hint=_parse_sdf_source_hint(sdf_path),
    )


def _res_name_from_hint(sdf_ligand: SDFLigand) -> str:
    if sdf_ligand.source_hint is not None and sdf_ligand.source_hint.res_name:
        return sdf_ligand.source_hint.res_name
    # Without an explicit residue hint, treat the ligand as generic so the
    # resolver creates a per-structure synthetic component ID instead of
    # guessing a CCD-like code from the filename.
    return "LIG"


def _format_element_symbol(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "X"
    if len(text) == 1:
        return text.upper()
    return text[0].upper() + text[1:].lower()


def _bond_order_and_aromatic(bond_type_id: int) -> tuple[str, str]:
    try:
        bond_type = BondType(bond_type_id)
    except ValueError:
        return "SING", "N"

    if bond_type == BondType.SINGLE:
        return "SING", "N"
    if bond_type == BondType.DOUBLE:
        return "DOUB", "N"
    if bond_type == BondType.TRIPLE:
        return "TRIP", "N"
    if bond_type == BondType.QUADRUPLE:
        return "QUAD", "N"
    if bond_type in (
        BondType.AROMATIC,
        BondType.AROMATIC_SINGLE,
        BondType.AROMATIC_DOUBLE,
        BondType.AROMATIC_TRIPLE,
    ):
        return "SING", "Y"
    return "SING", "N"


def _rdkit_bond_order_and_aromatic(bond) -> tuple[str, str]:
    from rdkit import Chem

    if bond.GetIsAromatic():
        return "SING", "Y"

    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        return "SING", "N"
    if bond_type == Chem.BondType.DOUBLE:
        return "DOUB", "N"
    if bond_type == Chem.BondType.TRIPLE:
        return "TRIP", "N"
    if bond_type == Chem.BondType.QUADRUPLE:
        return "QUAD", "N"
    return "SING", "N"


def _connected_component_count(
    atom_names: tuple[str, ...], bonds: tuple[tuple[str, str, str, str], ...]
) -> int:
    if not atom_names:
        return 0

    atom_index = {name: idx for idx, name in enumerate(atom_names)}
    adjacency: list[set[int]] = [set() for _ in atom_names]
    for atom_1, atom_2, _order, _aromatic in bonds:
        idx_1 = atom_index.get(atom_1)
        idx_2 = atom_index.get(atom_2)
        if idx_1 is None or idx_2 is None or idx_1 == idx_2:
            continue
        adjacency[idx_1].add(idx_2)
        adjacency[idx_2].add(idx_1)

    seen: set[int] = set()
    component_count = 0
    for start in range(len(atom_names)):
        if start in seen:
            continue
        component_count += 1
        stack = [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
    return component_count


def _add_explicit_hydrogens(component: SyntheticComponent) -> SyntheticComponent:
    if any(element in {"H", "D"} for element in component.atom_elements):
        return component

    try:
        from rdkit import Chem
    except Exception:
        return component

    atom_names = list(component.atom_names)
    atom_elements = list(component.atom_elements)
    atom_charges = [int(x) for x in component.atom_charges]
    atom_coords = [tuple(float(v) for v in xyz) for xyz in component.atom_coords]
    name_to_idx = {name: idx for idx, name in enumerate(atom_names)}

    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atom_names))
    for idx, (element, charge, coord) in enumerate(
        zip(atom_elements, atom_charges, atom_coords, strict=True)
    ):
        symbol = element if element != "X" else "*"
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(int(charge))
        mol.AddAtom(atom)
        conf.SetAtomPosition(idx, [float(coord[0]), float(coord[1]), float(coord[2])])

    bond_type_map = {
        "SING": Chem.BondType.SINGLE,
        "DOUB": Chem.BondType.DOUBLE,
        "TRIP": Chem.BondType.TRIPLE,
        "QUAD": Chem.BondType.QUADRUPLE,
    }
    for atom_1, atom_2, value_order, aromatic_flag in component.bonds:
        idx_1 = name_to_idx.get(atom_1)
        idx_2 = name_to_idx.get(atom_2)
        if idx_1 is None or idx_2 is None or idx_1 == idx_2:
            continue
        if aromatic_flag == "Y":
            bond_type = Chem.BondType.AROMATIC
        else:
            bond_type = bond_type_map.get(value_order, Chem.BondType.SINGLE)
        if mol.GetBondBetweenAtoms(int(idx_1), int(idx_2)) is None:
            mol.AddBond(int(idx_1), int(idx_2), bond_type)

    rdkit_mol = mol.GetMol()
    rdkit_mol.AddConformer(conf)
    try:
        Chem.SanitizeMol(rdkit_mol)
    except Exception:
        pass

    try:
        rdkit_mol = Chem.AddHs(rdkit_mol, addCoords=True)
    except Exception:
        return component

    conf = rdkit_mol.GetConformer()
    expanded_names = list(component.atom_names)
    for index in range(len(component.atom_names), rdkit_mol.GetNumAtoms()):
        expanded_names.append(f"H{index - len(component.atom_names) + 1}")

    expanded_elements: list[str] = []
    expanded_charges: list[int] = []
    expanded_coords: list[tuple[float, float, float]] = []
    for index, atom in enumerate(rdkit_mol.GetAtoms()):
        pos = conf.GetAtomPosition(index)
        expanded_elements.append(atom.GetSymbol())
        expanded_charges.append(int(atom.GetFormalCharge()))
        expanded_coords.append((float(pos.x), float(pos.y), float(pos.z)))

    expanded_bonds: list[tuple[str, str, str, str]] = []
    for bond in rdkit_mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        lo, hi = sorted((begin, end))
        value_order, aromatic_flag = _rdkit_bond_order_and_aromatic(bond)
        expanded_bonds.append(
            (expanded_names[lo], expanded_names[hi], value_order, aromatic_flag)
        )

    return SyntheticComponent(
        atom_names=tuple(expanded_names),
        atom_elements=tuple(expanded_elements),
        atom_charges=tuple(expanded_charges),
        atom_coords=tuple(expanded_coords),
        bonds=tuple(sorted(set(expanded_bonds))),
    )


def _chain_label_from_index(index: int) -> str:
    alphabet = string.ascii_uppercase
    label = ""
    value = index + 1
    while value > 0:
        value, remainder = divmod(value - 1, len(alphabet))
        label = alphabet[remainder] + label
    return label


def _next_available_chain_id(used_chain_ids: set[str]) -> str:
    candidate_index = 0
    while True:
        label = _chain_label_from_index(candidate_index)
        if label not in used_chain_ids:
            return label
        candidate_index += 1


def _ensure_atom_names(atom_array: AtomArray) -> np.ndarray:
    raw_names = [str(name).strip() for name in atom_array.atom_name]
    fallback_names = []
    for raw_name, element in zip(raw_names, atom_array.element, strict=False):
        base = re.sub(r"[^A-Za-z0-9]", "", raw_name).upper()
        if not base:
            base = _normalize_element(element)
        if not base:
            base = "X"
        fallback_names.append(base)

    counters: Counter[str] = Counter()
    unique_names: list[str] = []
    for base in fallback_names:
        counters[base] += 1
        if counters[base] == 1:
            unique_names.append(base)
        else:
            unique_names.append(f"{base}{counters[base]}")
    return np.asarray(unique_names, dtype=f"<U{max(len(name) for name in unique_names)}")


def _materialize_sdf_ligand(atom_array: AtomArray, residue_key: ResidueKey) -> AtomArray:
    ligand = atom_array.copy()
    n_atoms = ligand.array_length()
    ligand.chain_id = np.full(n_atoms, residue_key.chain_id, dtype=f"<U{max(1, len(residue_key.chain_id))}")
    ligand.res_id = np.full(n_atoms, residue_key.res_id, dtype=int)
    ligand.ins_code = np.full(n_atoms, residue_key.ins_code, dtype=f"<U{max(1, len(residue_key.ins_code))}")
    ligand.res_name = np.full(n_atoms, residue_key.res_name, dtype=f"<U{max(1, len(residue_key.res_name))}")
    ligand.hetero = np.ones(n_atoms, dtype=bool)
    ligand.atom_name = _ensure_atom_names(ligand)
    if "charge" not in ligand.get_annotation_categories():
        ligand.set_annotation("charge", np.zeros(n_atoms, dtype=int))
    else:
        ligand.set_annotation("charge", np.asarray(ligand.charge, dtype=int))
    return ligand


def _build_synthetic_component(atom_array: AtomArray) -> SyntheticComponent:
    atom_names = _ensure_atom_names(atom_array).tolist()
    atom_elements = [_format_element_symbol(element) for element in atom_array.element]
    atom_coords = [
        tuple(float(value) for value in atom_array.coord[index].tolist())
        for index in range(atom_array.array_length())
    ]
    if "charge" in atom_array.get_annotation_categories():
        atom_charges = [int(atom_array.charge[index]) for index in range(atom_array.array_length())]
    else:
        atom_charges = [0] * atom_array.array_length()

    bond_rows = (
        np.empty((0, 3), dtype=np.int64)
        if atom_array.bonds is None
        else np.asarray(atom_array.bonds.as_array(), dtype=np.int64)
    )
    seen_pairs: set[tuple[int, int]] = set()
    bonds: list[tuple[str, str, str, str]] = []
    for atom_1, atom_2, bond_type in bond_rows:
        atom_1 = int(atom_1)
        atom_2 = int(atom_2)
        if atom_1 == atom_2:
            continue
        pair = tuple(sorted((atom_1, atom_2)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        value_order, aromatic_flag = _bond_order_and_aromatic(int(bond_type))
        bonds.append(
            (
                atom_names[pair[0]],
                atom_names[pair[1]],
                value_order,
                aromatic_flag,
            )
        )

    component = SyntheticComponent(
        atom_names=tuple(atom_names),
        atom_elements=tuple(atom_elements),
        atom_charges=tuple(atom_charges),
        atom_coords=tuple(atom_coords),
        bonds=tuple(sorted(bonds)),
    )
    return _add_explicit_hydrogens(component)


def _build_synthetic_ccd_block(
    component_id: str,
    component: SyntheticComponent,
) -> CIFBlock:
    block = CIFBlock()
    atom_count = len(component.atom_names)
    atom_ordinals = np.arange(1, atom_count + 1, dtype=np.int64)
    coords = np.asarray(component.atom_coords, dtype=np.float64)
    atom_aromatic_flags = np.full(atom_count, "N", dtype=str)

    atom_index = {name: idx for idx, name in enumerate(component.atom_names)}
    for atom_1, atom_2, _value_order, aromatic_flag in component.bonds:
        if aromatic_flag != "Y":
            continue
        atom_aromatic_flags[atom_index[atom_1]] = "Y"
        atom_aromatic_flags[atom_index[atom_2]] = "Y"

    block["chem_comp"] = CIFCategory(
        {
            "id": np.array([component_id], dtype=str),
            "type": np.array(["non-polymer"], dtype=str),
            "name": np.array([component_id], dtype=str),
            "formula": np.array(["?"], dtype=str),
            "formula_weight": np.array(["0.0"], dtype=str),
            "pdbx_modified_date": np.array([date.today().isoformat()], dtype=str),
            "pdbx_release_status": np.array(["REL"], dtype=str),
            "pdbx_model_coordinates_missing_flag": np.array(["N"], dtype=str),
            "pdbx_ideal_coordinates_missing_flag": np.array(["N"], dtype=str),
        }
    )
    block["chem_comp_atom"] = CIFCategory(
        {
            "comp_id": np.array([component_id] * atom_count, dtype=str),
            "atom_id": np.array(component.atom_names, dtype=str),
            "alt_atom_id": np.array(component.atom_names, dtype=str),
            "type_symbol": np.array(component.atom_elements, dtype=str),
            "charge": np.array(component.atom_charges, dtype=str),
            "pdbx_align": np.array(["1"] * atom_count, dtype=str),
            "pdbx_aromatic_flag": atom_aromatic_flags,
            "pdbx_leaving_atom_flag": np.array(["N"] * atom_count, dtype=str),
            "pdbx_stereo_config": np.array(["N"] * atom_count, dtype=str),
            "model_Cartn_x": coords[:, 0],
            "model_Cartn_y": coords[:, 1],
            "model_Cartn_z": coords[:, 2],
            "pdbx_model_Cartn_x_ideal": coords[:, 0],
            "pdbx_model_Cartn_y_ideal": coords[:, 1],
            "pdbx_model_Cartn_z_ideal": coords[:, 2],
            "pdbx_component_atom_id": np.array(component.atom_names, dtype=str),
            "pdbx_component_comp_id": np.array([component_id] * atom_count, dtype=str),
            "pdbx_ordinal": atom_ordinals,
        }
    )

    if component.bonds:
        bond_count = len(component.bonds)
        block["chem_comp_bond"] = CIFCategory(
            {
                "comp_id": np.array([component_id] * bond_count, dtype=str),
                "atom_id_1": np.array([bond[0] for bond in component.bonds], dtype=str),
                "atom_id_2": np.array([bond[1] for bond in component.bonds], dtype=str),
                "value_order": np.array([bond[2] for bond in component.bonds], dtype=str),
                "pdbx_aromatic_flag": np.array([bond[3] for bond in component.bonds], dtype=str),
                "pdbx_stereo_config": np.array(["N"] * bond_count, dtype=str),
                "pdbx_ordinal": np.arange(1, bond_count + 1, dtype=np.int64),
            }
        )
    return block


def _component_validation_flags(component: SyntheticComponent) -> list[str]:
    flags: list[str] = []
    if len(component.atom_names) > 1 and not component.bonds:
        flags.append("empty_bond_table")
    connected_components = _connected_component_count(component.atom_names, component.bonds)
    if len(component.atom_names) > 1 and connected_components > 1:
        flags.append(f"disconnected_graph:{connected_components}")
    return flags

# This keeps track of the unique residues that are already present in the structure, 
# and assigns new residue keys to added ligands while ensuring no conflicts. 
# It also respects any source hints provided by the SDF file names when possible.
class ResidueAssigner:
    def __init__(
        self,
        used_chain_ids: set[str],
        used_residue_keys: set[ResidueKey],
    ):
        self.used_chain_ids = set(used_chain_ids)
        self.used_residue_keys = set(used_residue_keys)

    @classmethod
    def from_existing_atoms(
        cls,
        pdb_atoms: AtomArray,
        protein_residues: list[AtomArray],
    ) -> ResidueAssigner:
        return cls(
            used_chain_ids={str(chain_id).strip() for chain_id in pdb_atoms.chain_id.tolist()},
            used_residue_keys={
                _residue_key_from_array(residue)
                for residue in protein_residues
            },
        )

    def assign(self, sdf_ligand: SDFLigand, res_name: str) -> ResidueKey:
        if sdf_ligand.source_hint is not None:
            hinted_key = ResidueKey(
                chain_id=sdf_ligand.source_hint.chain_id,
                res_id=sdf_ligand.source_hint.res_id,
                ins_code=sdf_ligand.source_hint.ins_code,
                res_name=res_name,
            )
            if hinted_key not in self.used_residue_keys:
                self.used_residue_keys.add(hinted_key)
                self.used_chain_ids.add(hinted_key.chain_id)
                return hinted_key

        chain_id = _next_available_chain_id(self.used_chain_ids)
        self.used_chain_ids.add(chain_id)
        residue_key = ResidueKey(
            chain_id=chain_id,
            res_id=1,
            ins_code="",
            res_name=res_name,
        )
        self.used_residue_keys.add(residue_key)
        return residue_key


class LigandComponentResolver:
    def resolve(
        self,
        sdf_ligand: SDFLigand,
        structure_id: str,
        ligand_index: int,
    ) -> ComponentAssignment:
        source_res_name = _res_name_from_hint(sdf_ligand)
        if source_res_name in GENERIC_LIGAND_REFERENCE_IDS:
            return ComponentAssignment(
                component_id=f"{structure_id}_{ligand_index}",
                component_kind="synthetic",
                source_res_name=source_res_name,
            )
        return ComponentAssignment(
            component_id=source_res_name,
            component_kind="base",
            source_res_name=source_res_name,
        )


class SyntheticCCDBuilder:
    def __init__(self):
        self.synthetic_components: dict[str, SyntheticComponent] = {}

    @property
    def component_ids(self) -> list[str]:
        return sorted(self.synthetic_components)

    def register(self, component_id: str, atom_array: AtomArray) -> list[str]:
        component = _build_synthetic_component(atom_array)
        existing = self.synthetic_components.get(component_id)
        if existing is not None and existing != component:
            raise ValueError(
                f"Inconsistent synthetic component content for promoted ID {component_id}"
            )
        self.synthetic_components[component_id] = component
        return _component_validation_flags(component)

    def write(self, base_ccd_path: Path, out_path: Path) -> None:
        _write_augmented_ccd(
            base_ccd_path=base_ccd_path,
            augmented_ccd_out_path=out_path,
            synthetic_components=self.synthetic_components,
        )

# This takes in a structure and a set of ligands, and merges them into a single 
# structure with the ligands added as new residues.
class StructureMerger:
    def __init__(
        self,
        assigner_cls: type[ResidueAssigner],
        component_resolver: LigandComponentResolver,
        ccd_builder: SyntheticCCDBuilder,
    ):
        self.assigner_cls = assigner_cls
        self.component_resolver = component_resolver
        self.ccd_builder = ccd_builder

    def merge(
        self,
        pdb_path: str,
        sdf_paths: list[str],
        out_cif_path: str,
    ) -> MergeResult:
        from openfold3.core.data.io.structure.cif import write_structure

        pdb_path_obj = Path(pdb_path)
        out_cif_path_obj = Path(out_cif_path)
        out_cif_path_obj.parent.mkdir(parents=True, exist_ok=True)
        structure_id = pdb_path_obj.stem

        pdb_file = pdb.PDBFile.read(pdb_path_obj)
        pdb_atoms = pdb.get_structure(
            pdb_file,
            model=1,
            include_bonds=False,
            extra_fields=["charge"],
        )

        protein_residues = [
            residue.copy()
            for residue in _iter_residues(pdb_atoms)
            if _is_retained_protein_residue(residue)
        ]
        sdf_ligands = [
            _make_sdf_ligand(Path(sdf_path))
            for sdf_path in sdf_paths
        ]
        residue_assigner = self.assigner_cls.from_existing_atoms(pdb_atoms, protein_residues)

        added_ligands: list[AtomArray] = []
        promoted_component_ids: list[str] = []
        records: list[dict[str, Any]] = []
        added_records: list[dict[str, Any]] = []

        for ligand_index, sdf_ligand in enumerate(sdf_ligands, start=1):
            component_assignment = self.component_resolver.resolve(
                sdf_ligand=sdf_ligand,
                structure_id=structure_id,
                ligand_index=ligand_index,
            )
            assigned_residue = residue_assigner.assign(
                sdf_ligand,
                component_assignment.component_id,
            )
            added_ligand = _materialize_sdf_ligand(sdf_ligand.atom_array, assigned_residue)
            added_ligands.append(added_ligand)

            validation_flags: list[str] = []
            if component_assignment.component_kind == "synthetic":
                validation_flags = self.ccd_builder.register(
                    component_assignment.component_id,
                    sdf_ligand.atom_array,
                )
                promoted_component_ids.append(component_assignment.component_id)

            record: dict[str, Any] = {
                "sdf_path": str(sdf_ligand.path),
                "source_hint": (
                    None
                    if sdf_ligand.source_hint is None
                    else sdf_ligand.source_hint.to_dict()
                ),
                "action": "add",
                "source_res_name": component_assignment.source_res_name,
                "component_id": component_assignment.component_id,
                "component_kind": component_assignment.component_kind,
                "component_validation_flags": validation_flags,
                "assigned_residue": assigned_residue.to_dict(),
            }
            added_records.append(record)
            records.append(record)

        merged_chunks: list[AtomArray] = [*protein_residues, *added_ligands]
        if not merged_chunks:
            raise ValueError("Merged structure would be empty")

        merged_atoms = struc.concatenate(merged_chunks)
        merged_atoms = prepare_atom_array_for_of3_cif_write(merged_atoms)

        write_structure(
            atom_array=merged_atoms,
            output_path=out_cif_path_obj,
            data_block=sanitize_stem(pdb_path_obj.stem),
            include_bonds=True,
            make_ost_compatible=True,
        )

        release_date, experimental_method, resolution = extract_pdb_metadata(pdb_path_obj)
        _inject_required_cif_metadata(
            cif_path=out_cif_path_obj,
            atom_array=merged_atoms,
            release_date=release_date,
            experimental_method=experimental_method,
            resolution=resolution,
        )

        return MergeResult(
            summary={
                "counts": {
                    "added": len(added_records),
                    "promoted_components": len(promoted_component_ids),
                },
                "added": added_records,
                "records": records,
                "synthetic_component_ids": sorted(promoted_component_ids),
                "output_cif_path": str(out_cif_path_obj),
            },
            synthetic_components={
                component_id: self.ccd_builder.synthetic_components[component_id]
                for component_id in promoted_component_ids
            },
        )


class BatchMergeRunner:
    def __init__(
        self,
        merger: StructureMerger,
        ccd_builder: SyntheticCCDBuilder,
    ):
        self.merger = merger
        self.ccd_builder = ccd_builder

    def run(
        self,
        input_dir: str,
        output_dir: str,
        sample_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        input_dir_path = Path(input_dir)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        report_dir_path = _default_batch_report_dir(output_dir_path)
        report_dir_path.mkdir(parents=True, exist_ok=True)

        selected_sample_ids = None if sample_ids is None else set(sample_ids)
        sample_dirs = [
            path
            for path in sorted(input_dir_path.iterdir())
            if path.is_dir()
            and (selected_sample_ids is None or path.name in selected_sample_ids)
        ]
        if limit is not None:
            sample_dirs = sample_dirs[:limit]

        results: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []

        for sample_dir in tqdm(sample_dirs, desc="Merging PDB/SDF samples"):
            output_stem = sample_dir.name.lower()
            out_cif_path = output_dir_path / f"{output_stem}.cif"
            report_json_path = report_dir_path / f"{output_stem}.merge_report.json"
            try:
                merge_result = self.merger.merge(
                    pdb_path=str(_resolve_sample_pdb_path(sample_dir)),
                    sdf_paths=_resolve_sample_sdf_paths(sample_dir),
                    out_cif_path=str(out_cif_path),
                )
                summary = merge_result.summary
                summary["sample_id"] = sample_dir.name
                summary["sample_dir"] = str(sample_dir)
                report_json_path.write_text(json.dumps(summary, indent=2))
                results.append(summary)
            except Exception as exc:
                failure = {
                    "sample_id": sample_dir.name,
                    "sample_dir": str(sample_dir),
                    "error": str(exc),
                }
                failures.append(failure)
                report_json_path.write_text(
                    json.dumps(
                        {
                            "sample_id": sample_dir.name,
                            "sample_dir": str(sample_dir),
                            "status": "failed",
                            "error": str(exc),
                        },
                        indent=2,
                    )
                )

        return _summarize_batch_results(
            input_dir=input_dir_path,
            output_dir=output_dir_path,
            report_dir=report_dir_path,
            results=results,
            failures=failures,
            sample_count=len(sample_dirs),
            augmented_ccd_out_path=None,
            synthetic_components=self.ccd_builder.synthetic_components,
        )


def _is_retained_protein_residue(residue: AtomArray) -> bool:
    return bool(residue.array_length()) and not bool(residue.hetero[0])

def _default_reference_data_dir(parent_dir: Path) -> Path:
    return parent_dir / "reference_data"


def _default_base_ccd_path(parent_dir: Path) -> Path:
    return _default_reference_data_dir(parent_dir) / "components.cif"


def _default_augmented_ccd_path(parent_dir: Path) -> Path:
    return _default_reference_data_dir(parent_dir) / "components_harmonized.cif"


def _write_augmented_ccd(
    base_ccd_path: Path,
    augmented_ccd_out_path: Path,
    synthetic_components: dict[str, SyntheticComponent],
) -> None:
    base_ccd = pdbx.CIFFile.read(base_ccd_path)
    for component_id in sorted(synthetic_components):
        if component_id in base_ccd:
            raise ValueError(f"Promoted component ID already exists in CCD: {component_id}")
        base_ccd[component_id] = _build_synthetic_ccd_block(
            component_id,
            synthetic_components[component_id],
        )
    augmented_ccd_out_path.parent.mkdir(parents=True, exist_ok=True)
    base_ccd.write(augmented_ccd_out_path)


def merge_pdb_with_sdfs_to_cif(
    pdb_path: str,
    sdf_paths: list[str],
    out_cif_path: str,
) -> MergeResult:
    ccd_builder = SyntheticCCDBuilder()
    merger = StructureMerger(ResidueAssigner, LigandComponentResolver(), ccd_builder)
    return merger.merge(pdb_path=pdb_path, sdf_paths=sdf_paths, out_cif_path=out_cif_path)


def _collect_sdf_paths(explicit_paths: list[str], sdf_dir: str | None) -> list[str]:
    sdf_paths = [str(Path(path)) for path in explicit_paths]
    if sdf_dir is not None:
        sdf_paths.extend(str(path) for path in sorted(Path(sdf_dir).glob("*.sdf")))
    deduped: list[str] = []
    seen: set[str] = set()
    for path in sdf_paths:
        resolved = str(Path(path))
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    if not deduped:
        raise ValueError("At least one SDF path or --sdf-dir is required")
    return deduped


def _default_single_report_path(out_cif_path: Path) -> Path:
    return out_cif_path.parent / "merge_reports" / f"{out_cif_path.stem}.merge_report.json"


def _default_batch_report_dir(output_dir: Path) -> Path:
    return output_dir / "merge_reports"


def _summarize_batch_results(
    input_dir: Path,
    output_dir: Path,
    report_dir: Path,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    sample_count: int,
    augmented_ccd_out_path: Path | None,
    synthetic_components: dict[str, SyntheticComponent],
) -> dict[str, Any]:
    aggregate_counts = {"added": 0, "promoted_components": 0}
    for result in results:
        counts = result.get("counts", {})
        for key in aggregate_counts:
            aggregate_counts[key] += int(counts.get(key, 0))

    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "sample_count": sample_count,
        "success_count": len(results),
        "failure_count": len(failures),
        "aggregate_counts": aggregate_counts,
        "augmented_ccd_path": None if augmented_ccd_out_path is None else str(augmented_ccd_out_path),
        "synthetic_component_ids": sorted(synthetic_components),
        "failed_sample_ids": [failure["sample_id"] for failure in failures],
    }


def _resolve_sample_pdb_path(sample_dir: Path) -> Path:
    preferred = sample_dir / f"{sample_dir.name}.pdb"
    if preferred.exists():
        return preferred

    candidates = [
        path
        for path in sorted(sample_dir.glob("*.pdb"))
        if "_apo" not in path.stem and "_ligand" not in path.stem
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No mergeable PDB found in {sample_dir}")
    raise ValueError(
        f"Ambiguous PDB selection in {sample_dir}: {[path.name for path in candidates]}"
    )


def _resolve_sample_sdf_paths(sample_dir: Path) -> list[str]:
    preferred_ligand = sample_dir / f"{sample_dir.name}_ligand.sdf"
    if preferred_ligand.exists():
        return [str(preferred_ligand)]

    ligand_named = sorted(sample_dir.glob("*_ligand.sdf"))
    if len(ligand_named) == 1:
        return [str(ligand_named[0])]
    if len(ligand_named) > 1:
        raise ValueError(
            f"Ambiguous ligand SDF selection in {sample_dir}: {[path.name for path in ligand_named]}"
        )

    preferred = sample_dir / f"{sample_dir.name}.sdf"
    if preferred.exists():
        return [str(preferred)]

    all_sdfs = [
        str(path)
        for path in sorted(sample_dir.glob("*.sdf"))
        if not path.name.startswith(".")
    ]
    if all_sdfs:
        return all_sdfs

    raise FileNotFoundError(f"No SDF ligands found in {sample_dir}")


def merge_pdb_sdf_directory_to_cif_dir(
    input_dir: str,
    output_dir: str,
    base_ccd_path: str | None = None,
    augmented_ccd_out_path: str | None = None,
    ccd_download_url: str = DEFAULT_CCD_URL,
    sample_ids: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    resolved_base_ccd_path = (
        Path(base_ccd_path)
        if base_ccd_path is not None
        else _default_base_ccd_path(output_dir_path)
    )
    resolved_augmented_ccd_path = (
        Path(augmented_ccd_out_path)
        if augmented_ccd_out_path is not None
        else _default_augmented_ccd_path(output_dir_path)
    )
    ensure_ccd_file(resolved_base_ccd_path, ccd_download_url=ccd_download_url)
    ccd_builder = SyntheticCCDBuilder()
    merger = StructureMerger(ResidueAssigner, LigandComponentResolver(), ccd_builder)
    runner = BatchMergeRunner(merger, ccd_builder)
    summary = runner.run(
        input_dir=input_dir,
        output_dir=output_dir,
        sample_ids=sample_ids,
        limit=limit,
    )
    ccd_builder.write(resolved_base_ccd_path, resolved_augmented_ccd_path)
    summary["augmented_ccd_path"] = str(resolved_augmented_ccd_path)
    summary["synthetic_component_ids"] = ccd_builder.component_ids
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge a PDB structure with one or more SDF ligands into an OF3-compatible mmCIF.",
    )
    parser.add_argument("--pdb-path", help="Input PDB path")
    parser.add_argument(
        "--sdf-path",
        action="append",
        default=[],
        help="Input SDF path. Repeat for multiple ligands.",
    )
    parser.add_argument(
        "--sdf-dir",
        help="Directory containing ligand SDF files to merge",
    )
    parser.add_argument("--out-cif-path", help="Output mmCIF path")
    parser.add_argument(
        "--input-dir",
        help="Batch input directory containing one sample subdirectory per complex",
    )
    parser.add_argument(
        "--output-dir",
        help="Batch output directory for merged CIFs and per-sample reports",
    )
    parser.add_argument(
        "--base-ccd-path",
        help="Base CCD to extend. Defaults to <output-root>/reference_data/components.cif",
    )
    parser.add_argument(
        "--augmented-ccd-out-path",
        help=(
            "Path to write the augmented CCD. Defaults to "
            "<output-root>/reference_data/components_harmonized.cif"
        ),
    )
    parser.add_argument(
        "--ccd-download-url",
        default=DEFAULT_CCD_URL,
        help="URL used to download the base CCD when --base-ccd-path is missing",
    )
    parser.add_argument(
        "--report-json-path",
        help="Optional JSON path for the decision report. Defaults to <out-cif>.merge_report.json",
    )
    parser.add_argument(
        "--batch-report-json-path",
        help="Optional JSON path for the batch summary report. Defaults to <output-dir>/merge_reports/merge_batch_report.json",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Optional sample directory name to process. Repeat to process a subset in batch mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of sample directories processed in batch mode.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.input_dir is not None:
        if args.output_dir is None:
            raise ValueError("--output-dir is required when --input-dir is used")
        output_dir_path = Path(args.output_dir)
        resolved_base_ccd_path = (
            Path(args.base_ccd_path)
            if args.base_ccd_path is not None
            else _default_base_ccd_path(output_dir_path)
        )
        resolved_augmented_ccd_path = (
            Path(args.augmented_ccd_out_path)
            if args.augmented_ccd_out_path is not None
            else _default_augmented_ccd_path(output_dir_path)
        )
        ensure_ccd_file(resolved_base_ccd_path, ccd_download_url=args.ccd_download_url)
        ccd_builder = SyntheticCCDBuilder()
        merger = StructureMerger(
            ResidueAssigner,
            LigandComponentResolver(),
            ccd_builder,
        )
        runner = BatchMergeRunner(merger, ccd_builder)
        batch_summary = runner.run(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            sample_ids=args.sample_id or None,
            limit=args.limit,
        )
        ccd_builder.write(resolved_base_ccd_path, resolved_augmented_ccd_path)
        batch_summary["augmented_ccd_path"] = str(resolved_augmented_ccd_path)
        batch_summary["synthetic_component_ids"] = ccd_builder.component_ids
        batch_report_json_path = (
            Path(args.batch_report_json_path)
            if args.batch_report_json_path is not None
            else _default_batch_report_dir(output_dir_path) / "merge_batch_report.json"
        )
        batch_report_json_path.parent.mkdir(parents=True, exist_ok=True)
        batch_report_json_path.write_text(json.dumps(batch_summary, indent=2))
        print(
            "Merged "
            f"{batch_summary['success_count']}/{batch_summary['sample_count']} samples "
            f"to {batch_summary['output_dir']} "
            f"(failures={batch_summary['failure_count']}, reports={batch_summary['report_dir']})"
        )
        print(f"Augmented CCD: {batch_summary['augmented_ccd_path']}")
        print(f"Batch report: {batch_report_json_path}")
        return 0

    if args.pdb_path is None or args.out_cif_path is None:
        raise ValueError("Single-structure mode requires --pdb-path and --out-cif-path")

    sdf_paths = _collect_sdf_paths(args.sdf_path, args.sdf_dir)
    ccd_builder = SyntheticCCDBuilder()
    merger = StructureMerger(
        ResidueAssigner,
        LigandComponentResolver(),
        ccd_builder,
    )
    merge_result = merger.merge(
        pdb_path=args.pdb_path,
        sdf_paths=sdf_paths,
        out_cif_path=args.out_cif_path,
    )
    summary = merge_result.summary

    out_cif_path = Path(args.out_cif_path)
    single_root = out_cif_path.parent
    resolved_base_ccd_path = (
        Path(args.base_ccd_path)
        if args.base_ccd_path is not None
        else _default_base_ccd_path(single_root)
    )
    resolved_augmented_ccd_path = (
        Path(args.augmented_ccd_out_path)
        if args.augmented_ccd_out_path is not None
        else _default_augmented_ccd_path(single_root)
    )
    ensure_ccd_file(resolved_base_ccd_path, ccd_download_url=args.ccd_download_url)
    ccd_builder.write(resolved_base_ccd_path, resolved_augmented_ccd_path)
    summary["augmented_ccd_path"] = str(resolved_augmented_ccd_path)

    report_json_path = (
        Path(args.report_json_path)
        if args.report_json_path is not None
        else _default_single_report_path(out_cif_path)
    )
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(summary, indent=2))
    counts = summary["counts"]
    print(
        f"Merged CIF written to {summary['output_cif_path']} "
        f"(added={counts['added']}, promoted_components={counts['promoted_components']})"
    )
    print(f"Augmented CCD: {summary['augmented_ccd_path']}")
    print(f"Merge report: {report_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
