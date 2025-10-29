#!/usr/bin/env python3
"""Utilities for aligning rRNA chains from a PDB file to RiboVision references.

The script produces three outputs for a supplied PDB structure:

* A CSV map of helix labels to the nucleotide ranges observed in the PDB chains.
* A CSV that compares the reference helix ranges against the mapped PDB ranges.
* SVG depictions of the 23S (LSU) and 16S (SSU) secondary structures coloured by helix.

The implementation only relies on the Python standard library so it can run in
locked-down environments (such as the RiboVision deployment) without installing
additional packages.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "populate_db" / "Secondary_Structures" / "DATA"

SECONDARY_STRUCTURES_PATH = DATA_DIR / "SecondaryStructures.csv"
STRUCTURAL_DATA_PATH = DATA_DIR / "StructuralData2.csv"
STRUCTURE_DETAILS_PATH = DATA_DIR / "SecondaryStructureDetails.csv"

# The RiboVision 2.0 helix palette uses four discrete colours.  The precise values
# are not present in the data dumps that ship with the repository, so we mimic the
# published style with a distinct set of colours.
HELIX_COLORS = {
    "1": "#1f77b4",  # blue
    "2": "#ff7f0e",  # orange
    "3": "#2ca02c",  # green
    "4": "#d62728",  # red
}

# Ordering helper for insertion codes.
_INSERTION_ORDER = {"": 0}
_INSERTION_ORDER.update({chr(letter): idx + 1 for idx, letter in enumerate(range(ord("A"), ord("Z") + 1))})


@dataclass
class ReferenceNucleotide:
    """Single nucleotide entry in the reference secondary structure."""

    map_index: int
    res_num: int
    base: str
    x: float
    y: float
    helix: str
    helix_color: str


@dataclass
class ChainResidue:
    """Minimal representation of a residue extracted from a PDB chain."""

    resseq: int
    icode: str
    resname: str
    base: str

    @property
    def label(self) -> str:
        return f"{self.resseq}{self.icode}" if self.icode else str(self.resseq)

    @property
    def ordering_value(self) -> int:
        return self.resseq * 100 + _INSERTION_ORDER.get(self.icode, 0)


@dataclass
class AlignmentResult:
    aligned_reference: str
    aligned_chain: str
    score: int
    identity: float
    reference_to_chain: Dict[int, int]


class AlignmentError(RuntimeError):
    """Raised when no suitable chain can be aligned to the reference."""


def normalise_base(base: str) -> str:
    """Return a standardised base letter (A/C/G/U/N)."""

    base = base.strip().upper()
    if base == "T":
        return "U"
    if base in {"A", "C", "G", "U"}:
        return base
    return "N"


def canonical_base(resname: str) -> Optional[str]:
    """Convert a three-letter residue code into a canonical RNA base.

    The heuristic prefers known RNA bases and handles a range of modified
    residues that commonly occur in rRNA structures by selecting the first
    canonical base that appears in the residue name.
    """

    resname = resname.strip().upper()
    known = {
        "A": "A",
        "C": "C",
        "G": "G",
        "U": "U",
        "DA": "A",
        "DG": "G",
        "DC": "C",
        "DT": "U",
        "URA": "U",
        "URI": "U",
        "H2U": "U",
        "PSU": "U",
        "OMG": "G",
        "OMC": "C",
        "1MA": "A",
        "M2G": "G",
        "7MG": "G",
        "2MG": "G",
        "M7G": "G",
        "GMP": "G",
        "AMP": "A",
        "CMP": "C",
        "UMP": "U",
        "GTP": "G",
        "ATP": "A",
        "CTP": "C",
        "UTP": "U",
        "AET": "A",
        "I": "G",
        "INO": "G",
        "QUE": "G",
        "2MU": "U",
        "M2A": "A",
        "MIA": "A",
        "MSU": "U",
        "5MU": "U",
        "OMU": "U",
        "8MG": "G",
        "5MC": "C",
        "5MU": "U",
        "6HG": "G",
    }
    if resname in known:
        return known[resname]
    for token in ("A", "C", "G", "U", "T"):
        if token in resname:
            return "U" if token == "T" else token
    return None


def load_reference(ss_table: str) -> List[ReferenceNucleotide]:
    """Load the ordered reference nucleotides for the supplied secondary structure."""

    secondary_rows: Dict[int, ReferenceNucleotide] = {}
    with SECONDARY_STRUCTURES_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["SS_Table"] != ss_table:
                continue
            map_index = int(row["map_Index"])
            base = normalise_base(row["unModResName"])
            secondary_rows[map_index] = ReferenceNucleotide(
                map_index=map_index,
                res_num=int(row["resNum"]),
                base=base,
                x=float(row["X"]),
                y=float(row["Y"]),
                helix="",
                helix_color="1",
            )

    if not secondary_rows:
        raise AlignmentError(f"No secondary structure data available for {ss_table}.")

    with STRUCTURAL_DATA_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["SS_Table"] != ss_table:
                continue
            map_index = int(row["map_Index"])
            if map_index in secondary_rows:
                ref = secondary_rows[map_index]
                ref.helix = row["Helix_Num"] or "Unknown"
                ref.helix_color = row["Helix_Color"] or "1"

    ordered = sorted(secondary_rows.values(), key=lambda item: item.map_index)
    return ordered


def load_species_name(ss_table: str) -> str:
    """Return the species name for a secondary structure table identifier."""

    with STRUCTURE_DETAILS_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["SS_Table"] == ss_table:
                return row["Species_Name"]
    return ss_table


def parse_pdb(pdb_path: Path) -> Dict[str, List[ChainResidue]]:
    """Extract RNA chains from a PDB file."""

    chains: Dict[str, Dict[Tuple[int, str], ChainResidue]] = defaultdict(dict)
    with pdb_path.open() as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resname = line[17:20].strip()
            base = canonical_base(resname)
            if base is None:
                continue
            chain_id = line[21].strip() or "_"
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26].strip()
            key = (resseq, icode)
            # Preserve the first occurrence – it already captures the residue identity.
            if key not in chains[chain_id]:
                chains[chain_id][key] = ChainResidue(resseq, icode, resname, base)

    processed: Dict[str, List[ChainResidue]] = {}
    for chain_id, residues in chains.items():
        ordered = sorted(residues.values(), key=lambda item: (item.resseq, _INSERTION_ORDER.get(item.icode, 0)))
        if len(ordered) >= 50:
            processed[chain_id] = ordered
    return processed


def global_align(reference: Sequence[str], chain: Sequence[str]) -> AlignmentResult:
    """Needleman–Wunsch global alignment using integer arrays."""

    from array import array

    n, m = len(reference), len(chain)
    if n == 0 or m == 0:
        raise AlignmentError("Reference or chain sequence is empty; cannot align.")

    score_matrix = [array("i", [0] * (m + 1)) for _ in range(n + 1)]
    pointer_matrix = [array("b", [0] * (m + 1)) for _ in range(n + 1)]

    match_score, mismatch_score, gap_penalty = 2, -1, -2

    for i in range(1, n + 1):
        score_matrix[i][0] = score_matrix[i - 1][0] + gap_penalty
        pointer_matrix[i][0] = 1  # up
    for j in range(1, m + 1):
        score_matrix[0][j] = score_matrix[0][j - 1] + gap_penalty
        pointer_matrix[0][j] = 2  # left

    for i in range(1, n + 1):
        ref_char = reference[i - 1]
        for j in range(1, m + 1):
            chain_char = chain[j - 1]
            diag_score = score_matrix[i - 1][j - 1] + (match_score if ref_char == chain_char else mismatch_score)
            up_score = score_matrix[i - 1][j] + gap_penalty
            left_score = score_matrix[i][j - 1] + gap_penalty
            best_score = diag_score
            pointer = 0
            if up_score > best_score:
                best_score = up_score
                pointer = 1
            if left_score > best_score:
                best_score = left_score
                pointer = 2
            score_matrix[i][j] = best_score
            pointer_matrix[i][j] = pointer

    aligned_ref: List[str] = []
    aligned_chain: List[str] = []
    ref_index, chain_index = n, m
    ref_pos, chain_pos = n - 1, m - 1
    reference_to_chain: Dict[int, int] = {}

    while ref_index > 0 or chain_index > 0:
        if ref_index > 0 and chain_index > 0 and pointer_matrix[ref_index][chain_index] == 0:
            aligned_ref.append(reference[ref_index - 1])
            aligned_chain.append(chain[chain_index - 1])
            reference_to_chain[ref_pos] = chain_pos
            ref_index -= 1
            chain_index -= 1
            ref_pos -= 1
            chain_pos -= 1
        elif ref_index > 0 and (chain_index == 0 or pointer_matrix[ref_index][chain_index] == 1):
            aligned_ref.append(reference[ref_index - 1])
            aligned_chain.append("-")
            ref_index -= 1
            ref_pos -= 1
        else:
            aligned_ref.append("-")
            aligned_chain.append(chain[chain_index - 1])
            chain_index -= 1
            chain_pos -= 1

    aligned_ref.reverse()
    aligned_chain.reverse()

    matches = sum(1 for ref_char, chain_char in zip(aligned_ref, aligned_chain) if ref_char == chain_char and ref_char != "-")
    aligned_length = sum(1 for ref_char, chain_char in zip(aligned_ref, aligned_chain) if ref_char != "-" and chain_char != "-")
    identity = matches / aligned_length if aligned_length else 0.0

    return AlignmentResult(
        aligned_reference="".join(aligned_ref),
        aligned_chain="".join(aligned_chain),
        score=score_matrix[n][m],
        identity=identity,
        reference_to_chain=reference_to_chain,
    )


def select_best_chain(chains: Dict[str, List[ChainResidue]], reference: List[ReferenceNucleotide]) -> Tuple[str, AlignmentResult]:
    """Return the chain and alignment with the highest identity to the reference."""

    reference_sequence = [nuc.base for nuc in reference]
    best: Optional[Tuple[str, AlignmentResult]] = None
    for chain_id, residues in chains.items():
        chain_sequence = [res.base for res in residues]
        try:
            alignment = global_align(reference_sequence, chain_sequence)
        except AlignmentError:
            continue
        if best is None or alignment.identity > best[1].identity:
            best = (chain_id, alignment)
    if best is None:
        raise AlignmentError("Could not find a chain that aligns to the reference.")
    return best


def build_ranges(residues: Iterable[ChainResidue]) -> str:
    """Convert an ordered set of residues into a compact range string."""

    residue_list = list(residues)
    if not residue_list:
        return ""
    ranges: List[Tuple[ChainResidue, ChainResidue]] = []
    start = residue_list[0]
    end = residue_list[0]
    prev_value = start.ordering_value
    for residue in residue_list[1:]:
        value = residue.ordering_value
        if value - prev_value <= 100:
            end = residue
        else:
            ranges.append((start, end))
            start = end = residue
        prev_value = value
    ranges.append((start, end))
    formatted = []
    for start_res, end_res in ranges:
        if start_res.label == end_res.label:
            formatted.append(start_res.label)
        else:
            formatted.append(f"{start_res.label}-{end_res.label}")
    return "; ".join(formatted)


def build_reference_ranges(entries: Iterable[ReferenceNucleotide]) -> str:
    """Return a compact range representation for reference residues."""

    ordered = sorted(entries, key=lambda item: item.res_num)
    if not ordered:
        return ""
    ranges: List[Tuple[int, int]] = []
    start = ordered[0].res_num
    end = ordered[0].res_num
    for entry in ordered[1:]:
        if entry.res_num == end or entry.res_num == end + 1:
            end = entry.res_num
        else:
            ranges.append((start, end))
            start = end = entry.res_num
    ranges.append((start, end))
    formatted = []
    for start_val, end_val in ranges:
        if start_val == end_val:
            formatted.append(str(start_val))
        else:
            formatted.append(f"{start_val}-{end_val}")
    return "; ".join(formatted)


def summarise_helices(
    reference: List[ReferenceNucleotide],
    alignment: AlignmentResult,
    residues: List[ChainResidue],
    chain_id: str,
    rna_label: str,
    reference_name: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Build rows for the PDB helix map and the reference comparison CSVs."""

    helix_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, nucleotide in enumerate(reference):
        helix_indices[nucleotide.helix].append(idx)

    pdb_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []

    for helix_label, ref_indices in helix_indices.items():
        mapped_residues: List[ChainResidue] = []
        reference_entries = [reference[idx] for idx in ref_indices]
        for idx in ref_indices:
            chain_idx = alignment.reference_to_chain.get(idx)
            if chain_idx is not None and 0 <= chain_idx < len(residues):
                mapped_residues.append(residues[chain_idx])
        pdb_range = build_ranges(mapped_residues)
        reference_range = build_reference_ranges(reference_entries)
        pdb_rows.append(
            {
                "rna": rna_label,
                "reference": reference_name,
                "chain": chain_id,
                "helix": helix_label,
                "helix_color": HELIX_COLORS.get(reference_entries[0].helix_color, "#999999"),
                "mapped_range": pdb_range or "",
                "mapped_count": len(mapped_residues),
            }
        )
        comparison_rows.append(
            {
                "rna": rna_label,
                "reference": reference_name,
                "chain": chain_id,
                "helix": helix_label,
                "reference_range": reference_range,
                "reference_count": len(reference_entries),
                "pdb_range": pdb_range or "",
                "mapped_count": len(mapped_residues),
                "coverage": round(len(mapped_residues) / len(reference_entries), 3) if reference_entries else 0.0,
            }
        )
    return pdb_rows, comparison_rows


def create_secondary_structure_svg(
    reference: List[ReferenceNucleotide],
    output_path: Path,
    title: str,
):
    """Generate a simple SVG visualisation coloured by helix."""

    xs = [entry.x for entry in reference]
    ys = [entry.y for entry in reference]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    margin = 25
    width = max_x - min_x + 2 * margin
    height = max_y - min_y + 2 * margin

    def to_svg_coords(x: float, y: float) -> Tuple[float, float]:
        return x - min_x + margin, max_y - y + margin

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": f"{width:.2f}",
            "height": f"{height:.2f}",
            "viewBox": f"0 0 {width:.2f} {height:.2f}",
        },
    )
    ET.SubElement(svg, "title").text = title
    ET.SubElement(
        svg,
        "rect",
        {
            "x": "0",
            "y": "0",
            "width": f"{width:.2f}",
            "height": f"{height:.2f}",
            "fill": "#ffffff",
        },
    )

    # Backbone polyline to emphasise connectivity.
    points = []
    for entry in reference:
        x, y = to_svg_coords(entry.x, entry.y)
        points.append(f"{x:.2f},{y:.2f}")
    ET.SubElement(
        svg,
        "polyline",
        {
            "points": " ".join(points),
            "fill": "none",
            "stroke": "#cccccc",
            "stroke-width": "0.8",
        },
    )

    helix_points: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    for entry in reference:
        x, y = to_svg_coords(entry.x, entry.y)
        color = HELIX_COLORS.get(entry.helix_color, "#999999")
        ET.SubElement(
            svg,
            "circle",
            {
                "cx": f"{x:.2f}",
                "cy": f"{y:.2f}",
                "r": "1.6",
                "fill": color,
                "stroke": "none",
            },
        )
        helix_points[entry.helix].append((x, y, color))

    for helix_label, coords in helix_points.items():
        if not coords:
            continue
        avg_x = sum(point[0] for point in coords) / len(coords)
        avg_y = sum(point[1] for point in coords) / len(coords)
        color = coords[0][2]
        text = ET.SubElement(
            svg,
            "text",
            {
                "x": f"{avg_x:.2f}",
                "y": f"{avg_y:.2f}",
                "fill": color,
                "font-size": "7",
                "text-anchor": "middle",
                "font-family": "Arial, Helvetica, sans-serif",
            },
        )
        text.text = helix_label

    tree = ET.ElementTree(svg)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb", type=Path, help="Path to the PDB file to analyse")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rna_outputs"),
        help="Directory where CSV and SVG outputs will be written",
    )
    parser.add_argument("--lsu-table", default="ECOLI_LSU", help="Secondary structure table identifier for the LSU reference")
    parser.add_argument("--ssu-table", default="ECOLI_SSU", help="Secondary structure table identifier for the SSU reference")
    args = parser.parse_args()

    pdb_path: Path = args.pdb
    if not pdb_path.exists():
        raise SystemExit(f"PDB file not found: {pdb_path}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    chains = parse_pdb(pdb_path)
    if not chains:
        raise SystemExit("No RNA chains with at least 50 residues were found in the PDB file.")

    lsu_reference = load_reference(args.lsu_table)
    ssu_reference = load_reference(args.ssu_table)
    lsu_species = load_species_name(args.lsu_table)
    ssu_species = load_species_name(args.ssu_table)

    available_chains = dict(chains)
    lsu_chain_id, lsu_alignment = select_best_chain(available_chains, lsu_reference)
    lsu_residues = available_chains.pop(lsu_chain_id)
    print(
        f"Selected LSU chain {lsu_chain_id} (identity {lsu_alignment.identity:.2%}) aligned to {args.lsu_table} ({lsu_species})."
    )

    ssu_chain_id, ssu_alignment = select_best_chain(available_chains, ssu_reference)
    ssu_residues = available_chains[ssu_chain_id]
    print(
        f"Selected SSU chain {ssu_chain_id} (identity {ssu_alignment.identity:.2%}) aligned to {args.ssu_table} ({ssu_species})."
    )

    pdb_rows_lsu, comparison_rows_lsu = summarise_helices(
        lsu_reference, lsu_alignment, lsu_residues, lsu_chain_id, "23S LSU", lsu_species
    )
    pdb_rows_ssu, comparison_rows_ssu = summarise_helices(
        ssu_reference, ssu_alignment, ssu_residues, ssu_chain_id, "16S SSU", ssu_species
    )

    pdb_rows = pdb_rows_lsu + pdb_rows_ssu
    comparison_rows = comparison_rows_lsu + comparison_rows_ssu

    pdb_csv_path = output_dir / "pdb_helix_map.csv"
    comparison_csv_path = output_dir / "reference_comparison.csv"

    write_csv(
        pdb_csv_path,
        ["rna", "reference", "chain", "helix", "helix_color", "mapped_range", "mapped_count"],
        pdb_rows,
    )
    write_csv(
        comparison_csv_path,
        ["rna", "reference", "chain", "helix", "reference_range", "reference_count", "pdb_range", "mapped_count", "coverage"],
        comparison_rows,
    )

    create_secondary_structure_svg(
        lsu_reference,
        output_dir / "lsu_secondary_structure.svg",
        f"{args.lsu_table} secondary structure",
    )
    create_secondary_structure_svg(
        ssu_reference,
        output_dir / "ssu_secondary_structure.svg",
        f"{args.ssu_table} secondary structure",
    )

    print(f"Helix map written to: {pdb_csv_path}")
    print(f"Reference comparison written to: {comparison_csv_path}")
    print(f"SVGs saved to: {output_dir}")


if __name__ == "__main__":
    main()
