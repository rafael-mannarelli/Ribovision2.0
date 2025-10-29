"""Utility to align ribosomal RNA chains from a PDB/mmCIF structure against
RiboVision reference data and generate helix mapping CSVs and SVGs.

The script performs the following steps:

1. Loads 23S (LSU) and 16S (SSU) reference secondary structure data that ships
   with the repository.
2. Extracts RNA sequences from the provided structure file and automatically
   identifies the chains that correspond to the LSU and SSU.
3. Aligns the chain sequences to the best matching reference for each subunit
   and maps helices to structure residue ranges.
4. Writes two CSV files that describe the helix coverage in the structure and
   the corresponding reference regions.
5. Produces SVG renderings of the reference secondary structures with colored
   helix labels similar to the RiboVision visual style.

Example:
    python alignments/tools/pdb_alignment_outputs.py --pdb my_structure.cif \\
        --output-dir output/
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from Bio import pairwise2
from Bio.PDB import MMCIFParser, PDBParser

# Paths ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "populate_db" / "Secondary_Structures" / "DATA"
SECONDARY_STRUCTURES_CSV = DATA_DIR / "SecondaryStructures.csv"
STRUCTURAL_DATA_CSV = DATA_DIR / "StructuralData2.csv"
STRUCTURE_DETAILS_CSV = DATA_DIR / "SecondaryStructureDetails.csv"


# RNA residue normalisation -------------------------------------------------

RNA_BASE_MAP: Dict[str, str] = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
    "T": "U",
    "DA": "A",
    "DG": "G",
    "DT": "U",
    "DC": "C",
    "PSU": "U",
    "H2U": "U",
    "D": "U",
    "1MA": "A",
    "1MG": "G",
    "2MG": "G",
    "7MG": "G",
    "M2G": "G",
    "OMG": "G",
    "OMC": "C",
    "5MC": "C",
    "M5C": "C",
    "5MU": "U",
    "2MU": "U",
    "I": "A",
    "Y": "C",
    "HYP": "A",
    "QUE": "G",
    "G7M": "G",
}


@dataclass
class ReferenceStructure:
    """Container for a reference secondary structure entry."""

    ss_table: str
    species: str
    molecule: str  # "23S" or "16S"
    sequence: str
    residue_numbers: List[str]
    helix_labels: List[Optional[str]]
    coordinates: List[Tuple[float, float]]


@dataclass
class ChainSequence:
    """RNA chain extracted from the input structure."""

    chain_id: str
    sequence: str
    residue_labels: List[str]


@dataclass
class AlignmentResult:
    """Alignment of a structure chain to a reference."""

    chain: ChainSequence
    reference: ReferenceStructure
    score: float
    identity: float
    ref_alignment: str
    chain_alignment: str
    ref_to_chain: Dict[int, int]


# CSV loading ---------------------------------------------------------------

def normalise_base(name: str) -> str:
    """Convert a residue name to a canonical RNA base (A/C/G/U/N)."""

    clean = name.strip().upper()
    if clean in RNA_BASE_MAP:
        return RNA_BASE_MAP[clean]
    if len(clean) == 1 and clean in {"A", "C", "G", "U", "T"}:
        return "U" if clean == "T" else clean
    return "N"


def load_species_by_table() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with STRUCTURE_DETAILS_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[row["SS_Table"]] = row["Species_Name"]
    return mapping


def load_helix_assignments() -> Dict[str, Dict[int, str]]:
    assignments: Dict[str, Dict[int, str]] = {}
    with STRUCTURAL_DATA_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            table = row["SS_Table"]
            helix = row["Helix_Num"].strip()
            if not helix:
                continue
            assignments.setdefault(table, {})[int(row["map_Index"]) - 1] = helix
    return assignments


def load_reference_structures() -> Dict[str, List[ReferenceStructure]]:
    """Load 23S and 16S reference data from the CSV files."""

    species_by_table = load_species_by_table()
    helix_assignments = load_helix_assignments()
    grouped: Dict[str, Dict[int, Dict[str, object]]] = {"23S": {}, "16S": {}}

    with SECONDARY_STRUCTURES_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            molecule = row["molName"].strip()
            if molecule not in grouped:
                continue
            table = row["SS_Table"].strip()
            group = grouped[molecule].setdefault(
                table,
                {
                    "map_index": [],
                    "residue_numbers": [],
                    "bases": [],
                    "coords": [],
                },
            )
            map_index = int(row["map_Index"]) - 1
            group["map_index"].append(map_index)
            group["residue_numbers"].append(row["resNum"].strip())
            group["bases"].append(normalise_base(row["unModResName"]))
            group["coords"].append((float(row["X"]), float(row["Y"])))

    result: Dict[str, List[ReferenceStructure]] = {"23S": [], "16S": []}
    for molecule, tables in grouped.items():
        for table, payload in tables.items():
            order = sorted(range(len(payload["map_index"])), key=payload["map_index"].__getitem__)
            sequence = "".join(payload["bases"][i] for i in order)
            residue_numbers = [payload["residue_numbers"][i] for i in order]
            coords = [payload["coords"][i] for i in order]
            helix_labels: List[Optional[str]] = []
            table_assignments = helix_assignments.get(table, {})
            for idx in range(len(order)):
                helix_labels.append(table_assignments.get(payload["map_index"][order[idx]], None))
            result[molecule].append(
                ReferenceStructure(
                    ss_table=table,
                    species=species_by_table.get(table, "Unknown"),
                    molecule=molecule,
                    sequence=sequence,
                    residue_numbers=residue_numbers,
                    helix_labels=helix_labels,
                    coordinates=coords,
                )
            )
    return result


# Structure parsing ---------------------------------------------------------

def extract_rna_chains(structure_path: Path) -> List[ChainSequence]:
    """Parse the structure file and extract RNA chain sequences."""

    if structure_path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", str(structure_path))

    chains: List[ChainSequence] = []
    for model in structure:
        for chain in model:
            residues = []
            labels = []
            for res in chain:
                hetfield, resseq, icode = res.id
                if hetfield.strip():
                    continue
                base = normalise_base(res.resname)
                if base == "N":
                    # Skip residues that cannot be mapped to RNA bases.
                    continue
                residues.append(base)
                label = f"{chain.id}{resseq}{icode.strip()}".strip()
                labels.append(label)
            if residues:
                chains.append(ChainSequence(chain_id=chain.id, sequence="".join(residues), residue_labels=labels))
        break  # analyse first model only
    return chains


# Alignment helpers ---------------------------------------------------------

def align_sequences(reference: ReferenceStructure, chain: ChainSequence) -> AlignmentResult:
    alignment = pairwise2.align.globalms(
        reference.sequence,
        chain.sequence,
        2.0,
        -1.0,
        -5.0,
        -0.5,
        one_alignment_only=True,
        penalize_end_gaps=False,
    )
    if not alignment:
        raise ValueError("Failed to align sequences")
    ref_aln, chain_aln, score, *_ = alignment[0]
    matches = sum(1 for r, c in zip(ref_aln, chain_aln) if r == c and r != "-")
    identity = matches / max(1, min(len(reference.sequence), len(chain.sequence)))
    ref_to_chain: Dict[int, int] = {}
    ref_idx = -1
    chain_idx = -1
    for r_char, c_char in zip(ref_aln, chain_aln):
        if r_char != "-":
            ref_idx += 1
        if c_char != "-":
            chain_idx += 1
        if r_char != "-" and c_char != "-":
            ref_to_chain[ref_idx] = chain_idx
    return AlignmentResult(
        chain=chain,
        reference=reference,
        score=score,
        identity=identity,
        ref_alignment=ref_aln,
        chain_alignment=chain_aln,
        ref_to_chain=ref_to_chain,
    )


def pick_best_reference(references: Iterable[ReferenceStructure], chain: ChainSequence) -> AlignmentResult:
    best: Optional[AlignmentResult] = None
    for reference in references:
        result = align_sequences(reference, chain)
        if best is None or result.identity > best.identity:
            best = result
    if best is None:
        raise RuntimeError("Unable to align chain to any reference")
    return best


# Range utilities -----------------------------------------------------------

def build_segments(pairs: Sequence[Tuple[int, str]]) -> List[Tuple[str, str]]:
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda item: item[0])
    segments: List[Tuple[str, str]] = []
    start_idx, start_label = ordered[0]
    prev_idx, prev_label = start_idx, start_label
    for idx, label in ordered[1:]:
        if idx == prev_idx + 1:
            prev_idx = idx
            prev_label = label
            continue
        segments.append((start_label, prev_label))
        start_idx, start_label = idx, label
        prev_idx, prev_label = idx, label
    segments.append((start_label, prev_label))
    return segments


def format_segments(segments: Sequence[Tuple[str, str]]) -> str:
    if not segments:
        return ""
    formatted = []
    for start, end in segments:
        formatted.append(start if start == end else f"{start}-{end}")
    return "; ".join(formatted)


# Reporting -----------------------------------------------------------------

def summarise_alignment(result: AlignmentResult) -> Dict[str, object]:
    ref = result.reference
    chain = result.chain
    rows: List[Dict[str, object]] = []
    helix_indices: Dict[str, List[int]] = {}
    for idx, label in enumerate(ref.helix_labels):
        if not label:
            continue
        helix_indices.setdefault(label, []).append(idx)

    comparison_rows: List[Dict[str, object]] = []
    for helix_label, indices in sorted(helix_indices.items()):
        ref_pairs = [(i, ref.residue_numbers[i]) for i in indices]
        ref_segments = build_segments(ref_pairs)
        ref_segment_text = format_segments(ref_segments)
        mapped_pairs = [
            (result.ref_to_chain[i], chain.residue_labels[result.ref_to_chain[i]])
            for i in indices
            if i in result.ref_to_chain
        ]
        mapped_segments = build_segments(mapped_pairs)
        mapped_segment_text = format_segments(mapped_segments)
        rows.append(
            {
                "molecule": ref.molecule,
                "species": ref.species,
                "reference_table": ref.ss_table,
                "helix_label": helix_label,
                "pdb_chain": chain.chain_id,
                "pdb_ranges": mapped_segment_text,
                "mapped_residue_count": len(mapped_pairs),
            }
        )
        comparison_rows.append(
            {
                "molecule": ref.molecule,
                "species": ref.species,
                "reference_table": ref.ss_table,
                "helix_label": helix_label,
                "pdb_chain": chain.chain_id,
                "reference_ranges": ref_segment_text,
                "reference_length": len(indices),
                "pdb_ranges": mapped_segment_text,
                "mapped_residue_count": len(mapped_pairs),
                "missing_reference_positions": len(indices) - len(mapped_pairs),
            }
        )
    return {"elements": rows, "comparison": comparison_rows}


# SVG generation ------------------------------------------------------------

def colour_for_label(label: str, total: int, index: int) -> str:
    hue = (index / max(total, 1)) * 360.0
    return f"hsl({hue:.0f}, 70%, 50%)"


def create_secondary_structure_svg(reference: ReferenceStructure, output_path: Path) -> None:
    coords = reference.coordinates
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    margin = 20.0
    width = max_x - min_x + 2 * margin
    height = max_y - min_y + 2 * margin

    helix_indices: Dict[str, List[int]] = {}
    for idx, label in enumerate(reference.helix_labels):
        if label:
            helix_indices.setdefault(label, []).append(idx)

    sorted_labels = sorted(helix_indices)
    colour_map = {
        label: colour_for_label(label, len(sorted_labels), i)
        for i, label in enumerate(sorted_labels)
    }

    def transform(point: Tuple[float, float]) -> Tuple[float, float]:
        x, y = point
        return x - min_x + margin, height - (y - min_y + margin)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.1f}" height="{height:.1f}" '
            f'viewBox="0 0 {width:.1f} {height:.1f}">\n'
        )
        handle.write(
            "  <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\" stroke=\"#cccccc\" stroke-width=\"0.5\"/>\n"
        )
        handle.write(
            f"  <text x=\"{width / 2:.1f}\" y=\"24\" text-anchor=\"middle\" "
            "font-family=\"Arial, sans-serif\" font-size=\"16\" fill=\"#333333\">"
            f"{reference.species} {reference.molecule} ({reference.ss_table})" "</text>\n"
        )
        for idx, (x, y) in enumerate(coords):
            label = reference.helix_labels[idx]
            colour = colour_map.get(label, "#b0b0b0")
            tx, ty = transform((x, y))
            handle.write(
                f"  <circle cx=\"{tx:.2f}\" cy=\"{ty:.2f}\" r=\"1.4\" fill=\"{colour}\" stroke=\"#333\" stroke-width=\"0.2\"/>\n"
            )
        for label, indices in helix_indices.items():
            hx = sum(coords[i][0] for i in indices) / len(indices)
            hy = sum(coords[i][1] for i in indices) / len(indices)
            tx, ty = transform((hx, hy))
            handle.write(
                f"  <text x=\"{tx:.2f}\" y=\"{ty:.2f}\" text-anchor=\"middle\" "
                "font-family=\"Arial, sans-serif\" font-size=\"6\" fill=\"#000000\">"
                f"{label}" "</text>\n"
            )
        handle.write("</svg>\n")


# CLI -----------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate helix mapping artefacts for ribosomal structures.")
    parser.add_argument("--pdb", required=True, help="Path to the input PDB or mmCIF file.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where CSV and SVG artefacts will be written (default: output).",
    )
    args = parser.parse_args(argv)

    pdb_path = Path(args.pdb).expanduser().resolve()
    if not pdb_path.exists():
        raise SystemExit(f"Structure file not found: {pdb_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    references = load_reference_structures()
    chains = extract_rna_chains(pdb_path)
    if not chains:
        raise SystemExit("No RNA chains were detected in the provided structure.")

    lsu_result: Optional[AlignmentResult] = None
    ssu_result: Optional[AlignmentResult] = None

    if len(chains) == 1:
        raise SystemExit("A complete ribosome should contain at least two RNA chains (LSU and SSU).")

    remaining_chains = chains.copy()
    lsu_candidates = [pick_best_reference(references["23S"], chain) for chain in remaining_chains]
    lsu_result = max(lsu_candidates, key=lambda res: res.identity)
    remaining_chains = [chain for chain in remaining_chains if chain.chain_id != lsu_result.chain.chain_id]

    if not remaining_chains:
        raise SystemExit("Unable to identify a second RNA chain for the SSU.")

    ssu_candidates = [pick_best_reference(references["16S"], chain) for chain in remaining_chains]
    ssu_result = max(ssu_candidates, key=lambda res: res.identity)

    lsu_summary = summarise_alignment(lsu_result)
    ssu_summary = summarise_alignment(ssu_result)

    elements_csv = output_dir / "helix_elements_map.csv"
    comparison_csv = output_dir / "helix_reference_comparison.csv"

    with elements_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "molecule",
            "species",
            "reference_table",
            "helix_label",
            "pdb_chain",
            "pdb_ranges",
            "mapped_residue_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in lsu_summary["elements"] + ssu_summary["elements"]:
            writer.writerow(row)

    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "molecule",
            "species",
            "reference_table",
            "helix_label",
            "pdb_chain",
            "reference_ranges",
            "reference_length",
            "pdb_ranges",
            "mapped_residue_count",
            "missing_reference_positions",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in lsu_summary["comparison"] + ssu_summary["comparison"]:
            writer.writerow(row)

    lsu_svg = output_dir / "lsu_secondary_structure.svg"
    ssu_svg = output_dir / "ssu_secondary_structure.svg"
    create_secondary_structure_svg(lsu_result.reference, lsu_svg)
    create_secondary_structure_svg(ssu_result.reference, ssu_svg)

    print("Selected references:")
    print(
        f"  LSU: {lsu_result.reference.species} ({lsu_result.reference.ss_table}) - "
        f"chain {lsu_result.chain.chain_id}, identity {lsu_result.identity:.2%}"
    )
    print(
        f"  SSU: {ssu_result.reference.species} ({ssu_result.reference.ss_table}) - "
        f"chain {ssu_result.chain.chain_id}, identity {ssu_result.identity:.2%}"
    )
    print(f"Helix element map written to: {elements_csv}")
    print(f"Reference comparison written to: {comparison_csv}")
    print(f"LSU secondary structure SVG: {lsu_svg}")
    print(f"SSU secondary structure SVG: {ssu_svg}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
