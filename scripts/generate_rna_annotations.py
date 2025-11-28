#!/usr/bin/env python3
"""Utility to align an rRNA structure to RiboVision references and export annotations."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from Bio import pairwise2
    from Bio.PDB import MMCIFParser, PDBParser
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "Biopython is required to run this script. Install it with 'pip install biopython'."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "populate_db" / "Secondary_Structures" / "DATA"

NUCLEOTIDE_MAP: Dict[str, str] = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
    "DA": "A",
    "DC": "C",
    "DG": "G",
    "DT": "U",
    "ADE": "A",
    "CYT": "C",
    "GUA": "G",
    "URI": "U",
    "PSU": "U",
    "M2G": "G",
    "OMG": "G",
    "1MA": "A",
    "MIA": "A",
    "5MC": "C",
    "OMC": "C",
    "5MU": "U",
    "2MU": "U",
    "H2U": "U",
    "7MG": "G",
    "YYG": "G",
}

HELIX_PALETTE = {
    "0": "#B9B9B9",
    "1": "#0D72BA",
    "2": "#EF4136",
    "3": "#FDB913",
    "4": "#39B54A",
}


@dataclass
class ResidueRecord:
    map_index: int
    residue: str
    resnum: str
    x: float
    y: float
    helix: Optional[str]
    helix_color: str


@dataclass
class ReferenceSequence:
    ss_table: str
    mol_name: str
    species: str
    residues: List[ResidueRecord]

    @property
    def sequence(self) -> str:
        return "".join(r.residue for r in self.residues)

    @property
    def resnums(self) -> List[str]:
        return [r.resnum for r in self.residues]


@dataclass
class ChainSequence:
    chain_id: str
    sequence: str
    residue_labels: List[str]
    bio_residues: List


def load_species_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    details_path = DATA_DIR / "SecondaryStructureDetails.csv"
    with details_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lookup[row["SS_Table"]] = row["Species_Name"]
    return lookup


def load_residue_records() -> Dict[Tuple[str, str], List[ResidueRecord]]:
    helix_lookup: Dict[str, Dict[int, Tuple[str, str]]] = defaultdict(dict)
    struct_data_path = DATA_DIR / "StructuralData2.csv"
    with struct_data_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                map_index = int(row["map_Index"])
            except (TypeError, ValueError):
                continue
            helix_raw = (row.get("Helix_Num", "") or "").strip()
            helix_num = helix_raw if helix_raw and helix_raw != "0" else None
            helix_color = row.get("Helix_Color", "0") or "0"
            helix_lookup[row["SS_Table"]][map_index] = (helix_num, helix_color)

    result: Dict[Tuple[str, str], List[ResidueRecord]] = defaultdict(list)
    sec_struct_path = DATA_DIR / "SecondaryStructures.csv"
    with sec_struct_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mol = row["molName"].strip()
            if mol not in {"23S", "16S"}:
                continue
            ss_table = row["SS_Table"].strip()
            try:
                map_index = int(row["map_Index"])
                x = float(row["X"])
                y = float(row["Y"])
            except (TypeError, ValueError):
                continue
            residue = (row["unModResName"] or "").strip().upper()
            if residue not in {"A", "C", "G", "U"}:
                continue
            helix_num, helix_color = helix_lookup.get(ss_table, {}).get(map_index, (None, "0"))
            record = ResidueRecord(
                map_index=map_index,
                residue=residue,
                resnum=row["resNum"].strip(),
                x=x,
                y=y,
                helix=helix_num,
                helix_color=helix_color,
            )
            result[(ss_table, mol)].append(record)

    for records in result.values():
        records.sort(key=lambda r: r.map_index)
    return result


def build_reference_sequences() -> Dict[Tuple[str, str], ReferenceSequence]:
    species_lookup = load_species_lookup()
    residue_records = load_residue_records()
    references: Dict[Tuple[str, str], ReferenceSequence] = {}
    for (ss_table, mol), records in residue_records.items():
        species = species_lookup.get(ss_table, "Unknown")
        references[(ss_table, mol)] = ReferenceSequence(
            ss_table=ss_table,
            mol_name=mol,
            species=species,
            residues=records,
        )
    return references


def iter_structure_chains(structure_path: Path) -> Iterable[ChainSequence]:
    parser = MMCIFParser(QUIET=True) if structure_path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
    structure = parser.get_structure("structure", str(structure_path))

    for model in structure:
        for chain in model:
            residues = []
            labels = []
            bio_residues = []
            for residue in chain:
                resname = residue.get_resname().strip().upper()
                base = residue_to_base(resname)
                if base is None:
                    continue
                residues.append(base)
                label = residue_label(residue)
                labels.append(label)
                bio_residues.append(residue)
            if residues:
                yield ChainSequence(
                    chain_id=chain.id,
                    sequence="".join(residues),
                    residue_labels=labels,
                    bio_residues=bio_residues,
                )
        break  # only first model


def residue_to_base(resname: str) -> Optional[str]:
    if resname in NUCLEOTIDE_MAP:
        return NUCLEOTIDE_MAP[resname]
    if len(resname) == 1 and resname in {"A", "C", "G", "U"}:
        return resname
    if resname.endswith("A"):
        return "A"
    if resname.endswith("C"):
        return "C"
    if resname.endswith("G"):
        return "G"
    if resname.endswith("U") or resname.endswith("T"):
        return "U"
    return None


def residue_label(residue) -> str:
    resseq = residue.id[1]
    icode = residue.id[2].strip() or ""
    return f"{resseq}{icode}"


@dataclass
class AlignmentResult:
    reference: ReferenceSequence
    chain: ChainSequence
    ref_to_chain: Dict[int, int]
    identity: float
    coverage: float


def align_sequences(reference: ReferenceSequence, chain: ChainSequence) -> AlignmentResult:
    # Use a semi-global alignment that does not penalise terminal gaps on the
    # structure chain. Experimental or truncated structures frequently miss
    # nucleotides at the ends, and penalising those gaps can force suboptimal
    # alignments that shift the entire mapping.
    alignment = pairwise2.align.globalms(
        reference.sequence,
        chain.sequence,
        2.0,
        -1.0,
        -5.0,
        -1.0,
        one_alignment_only=True,
        penalize_end_gaps=(True, False),
    )[0]
    ref_aln, chain_aln, score, _, _ = alignment

    ref_pos = 0
    chain_pos = 0
    ref_to_chain: Dict[int, int] = {}
    matches = 0
    aligned_pairs = 0
    for r_char, c_char in zip(ref_aln, chain_aln):
        if r_char != "-":
            ref_pos += 1
        if c_char != "-":
            chain_pos += 1
        if r_char != "-" and c_char != "-":
            ref_to_chain[ref_pos - 1] = chain_pos - 1
            aligned_pairs += 1
            if r_char == c_char:
                matches += 1
    identity = matches / aligned_pairs if aligned_pairs else 0.0
    coverage = aligned_pairs / len(reference.residues) if reference.residues else 0.0
    return AlignmentResult(reference=reference, chain=chain, ref_to_chain=ref_to_chain, identity=identity, coverage=coverage)


def pick_best_alignments(
    references: Dict[Tuple[str, str], ReferenceSequence], chains: Iterable[ChainSequence]
) -> Dict[str, AlignmentResult]:
    """Pick best reference alignments, preferring matching species for 23S.

    The 23S and 16S references should come from the same organism as the input
    structure. We first find the best 16S alignment and then, if possible,
    select the 23S reference from the same species. This avoids pairing the 23S
    chain with an unrelated species when another species aligns slightly
    better.
    """

    by_mol: Dict[str, List[AlignmentResult]] = defaultdict(list)

    for chain in chains:
        for (ss_table, mol), reference in references.items():
            if mol not in {"23S", "16S"}:
                continue
            result = align_sequences(reference, chain)
            by_mol[mol].append(result)

    best: Dict[str, AlignmentResult] = {}

    def pick_best(results: List[AlignmentResult]) -> Optional[AlignmentResult]:
        if not results:
            return None
        # Prioritise alignments that cover more of the reference while still
        # maximising identity. This avoids selecting a short, high-identity
        # fragment over a more comprehensive 23S match.
        return max(results, key=lambda r: (r.identity * r.coverage, r.identity))

    best_16s = pick_best(by_mol.get("16S", []))
    if best_16s:
        best["16S"] = best_16s

    preferred_species = best_16s.reference.species if best_16s else None

    def pick_best_23s() -> Optional[AlignmentResult]:
        results = by_mol.get("23S", [])
        if preferred_species:
            matching_species = [r for r in results if r.reference.species == preferred_species]
            selected = pick_best(matching_species)
            if selected:
                return selected
        return pick_best(results)

    best_23s = pick_best_23s()
    if best_23s:
        best["23S"] = best_23s

    return best


def group_consecutive(indices: Sequence[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    sorted_idx = sorted(indices)
    groups = []
    start = sorted_idx[0]
    prev = start
    for idx in sorted_idx[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = prev = idx
    groups.append((start, prev))
    return groups


def format_ranges(indices: Sequence[int], labels: Sequence[str]) -> str:
    if not indices:
        return ""
    ranges = []
    for start, end in group_consecutive(indices):
        start_label = labels[start]
        end_label = labels[end]
        if start == end:
            ranges.append(str(start_label))
        else:
            ranges.append(f"{start_label}-{end_label}")
    return ";".join(ranges)


def summarise_helices(result: AlignmentResult) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, str]]:
    reference = result.reference
    chain = result.chain
    ref_indices_by_helix: Dict[str, List[int]] = defaultdict(list)
    helix_colors: Dict[str, List[str]] = defaultdict(list)

    for idx, residue in enumerate(reference.residues):
        if not residue.helix:
            continue
        ref_indices_by_helix[residue.helix].append(idx)
        helix_colors[residue.helix].append(residue.helix_color)

    elements_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []
    helix_palette: Dict[str, str] = {}

    for helix, indices in sorted(ref_indices_by_helix.items(), key=lambda kv: kv[0]):
        ref_labels = [reference.residues[i].resnum for i in range(len(reference.residues))]
        ref_range = format_ranges(indices, ref_labels)

        chain_positions = [result.ref_to_chain.get(i) for i in indices]
        mapped_positions = [p for p in chain_positions if p is not None]
        pdb_range = format_ranges(mapped_positions, chain.residue_labels) if mapped_positions else ""

        elements_rows.append(
            {
                "helix_label": helix,
                "pdb_chain": chain.chain_id,
                "pdb_ranges": pdb_range,
                "mapped_residues": len(mapped_positions),
            }
        )

        comparison_rows.append(
            {
                "helix_label": helix,
                "reference_dataset": reference.ss_table,
                "reference_species": reference.species,
                "reference_range": ref_range,
                "reference_length": len(indices),
                "pdb_chain": chain.chain_id,
                "pdb_ranges": pdb_range,
                "pdb_mapped_length": len(mapped_positions),
                "pct_mapped": round(len(mapped_positions) / len(indices), 3) if indices else 0.0,
                "alignment_identity": round(result.identity, 3),
                "alignment_coverage": round(result.coverage, 3),
            }
        )

        color_votes = Counter(helix_colors[helix])
        color_key = color_votes.most_common(1)[0][0] if color_votes else "0"
        helix_palette[helix] = HELIX_PALETTE.get(color_key, HELIX_PALETTE["0"])

    return elements_rows, comparison_rows, helix_palette


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_svg(result: AlignmentResult, helix_palette: Dict[str, str], output_path: Path) -> None:
    reference = result.reference
    if not reference.residues:
        return

    xs = [r.x for r in reference.residues]
    ys = [r.y for r in reference.residues]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x + 40
    height = max_y - min_y + 40

    font_size = 12.0
    circle_radius = 1.5
    details_path = DATA_DIR / "SecondaryStructureDetails.csv"
    with details_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["SS_Table"].strip() == reference.ss_table:
                try:
                    font_size = float(row.get("Font_Size_SVG", 12))
                    circle_radius = float(row.get("Circle_Radius", 1.5))
                except (TypeError, ValueError):
                    pass
                break

    header = f"<svg xmlns='http://www.w3.org/2000/svg' width='{width:.1f}' height='{height:.1f}' viewBox='{min_x-20:.1f} {min_y-20:.1f} {width:.1f} {height:.1f}'>"
    elements = [header, f"<title>{reference.species} {reference.mol_name} ({reference.ss_table})</title>"]

    for residue in reference.residues:
        color = HELIX_PALETTE.get(residue.helix_color, HELIX_PALETTE["0"])
        elements.append(
            f"<circle cx='{residue.x:.3f}' cy='{residue.y:.3f}' r='{circle_radius:.3f}' fill='{color}' stroke='#1f1f1f' stroke-width='0.4'/>"
        )

    sums: Dict[str, Tuple[float, float, int]] = {}
    for residue in reference.residues:
        if not residue.helix:
            continue
        sx, sy, count = sums.get(residue.helix, (0.0, 0.0, 0))
        sums[residue.helix] = (sx + residue.x, sy + residue.y, count + 1)

    for helix, (sx, sy, count) in sums.items():
        if not count:
            continue
        color = helix_palette.get(helix, HELIX_PALETTE["0"])
        label_x = sx / count
        label_y = sy / count
        elements.append(
            f"<text x='{label_x:.3f}' y='{label_y:.3f}' font-size='{font_size}' fill='{color}' text-anchor='middle'>{helix}</text>"
        )

    elements.append("</svg>")
    output_path.write_text("\n".join(elements))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb", type=Path, help="Path to the input PDB or mmCIF file")
    parser.add_argument("--output", type=Path, default=Path.cwd(), help="Directory to store generated files")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        raise SystemExit(f"Secondary structure data directory not found: {DATA_DIR}")
    if not args.pdb.exists():
        raise SystemExit(f"Input structure not found: {args.pdb}")

    references = build_reference_sequences()
    chains = list(iter_structure_chains(args.pdb))
    if not chains:
        raise SystemExit("No RNA chains with recognised nucleotides were found in the structure.")

    best = pick_best_alignments(references, chains)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    for mol in ("23S", "16S"):
        result = best.get(mol)
        if not result:
            print(f"No suitable reference found for {mol}.")
            continue
        elements_rows, comparison_rows, helix_palette = summarise_helices(result)
        base_name = f"{args.pdb.stem}_{mol}".lower()
        elements_path = output_dir / f"{base_name}_elements.csv"
        comparison_path = output_dir / f"{base_name}_reference_comparison.csv"
        svg_path = output_dir / f"{base_name}_secondary_structure.svg"

        write_csv(
            elements_path,
            elements_rows,
            ["helix_label", "pdb_chain", "pdb_ranges", "mapped_residues"],
        )
        write_csv(
            comparison_path,
            comparison_rows,
            [
                "helix_label",
                "reference_dataset",
                "reference_species",
                "reference_range",
                "reference_length",
                "pdb_chain",
                "pdb_ranges",
                "pdb_mapped_length",
                "pct_mapped",
                "alignment_identity",
                "alignment_coverage",
            ],
        )
        generate_svg(result, helix_palette, svg_path)

        print(
            f"{mol}: matched chain {result.chain.chain_id} to {result.reference.ss_table}"
            f" ({result.reference.species}) with identity {result.identity:.3f}"
        )
        print(f"  Elements map: {elements_path}")
        print(f"  Reference comparison: {comparison_path}")
        print(f"  Secondary structure SVG: {svg_path}")


if __name__ == "__main__":
    main()
