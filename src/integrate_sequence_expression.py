#!/usr/bin/env python3
"""
CLI driver for integrating promoter/5' UTR FASTA sequences with averaged VST expression data.

Merges a 2D VST expression matrix with promoter and stitched 5' UTR sequences to produce
a flattened 1D TSV dataset suitable for Mechanistic Interpretability training.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import pandas as pd

SequenceMap = dict[str, str]
UtrBlockMap = dict[str, list[tuple[int, str]]]

FBgn_PATTERN = re.compile(r"(FBgn\d+)")
UTR_HEADER_PATTERN = re.compile(r"(FBgn\d+)_block_(\d+)")


def open_maybe_gzip(path: Path) -> TextIO:
    """Open a plain-text or gzip-compressed file for reading as text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def iter_fasta_records(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (header, sequence) pairs from a FASTA file without loading all records at once."""
    current_header: str | None = None
    sequence_parts: list[str] = []
    record_count = 0

    with open_maybe_gzip(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    record_count += 1
                    yield current_header, "".join(sequence_parts).upper()
                current_header = line[1:].strip()
                sequence_parts = []
            else:
                sequence_parts.append(line)

    if current_header is not None:
        record_count += 1
        yield current_header, "".join(sequence_parts).upper()

    if record_count == 0:
        raise ValueError(f"No FASTA records found in {path}")


def extract_gene_id(header: str) -> str:
    """Extract the FBgn gene ID from a promoter FASTA header."""
    first_token = header.split(None, 1)[0]
    match = FBgn_PATTERN.search(first_token)
    if match:
        return match.group(1)
    return first_token


def parse_utr_header(header: str) -> tuple[str, int]:
    """Parse a 5' UTR exon FASTA header into gene ID and block number."""
    first_token = header.split(None, 1)[0]
    match = UTR_HEADER_PATTERN.match(first_token)
    if not match:
        raise ValueError(
            f"UTR FASTA header does not contain '<FBgn>_block_<N>': {header}"
        )
    return match.group(1), int(match.group(2))


def load_promoter_map(promoters_path: Path) -> SequenceMap:
    """Load promoter sequences keyed by gene ID."""
    promoter_map: SequenceMap = {}
    for header, sequence in iter_fasta_records(promoters_path):
        gene_id = extract_gene_id(header)
        if gene_id in promoter_map:
            raise ValueError(f"Duplicate promoter sequence for gene {gene_id}")
        promoter_map[gene_id] = sequence

    if not promoter_map:
        raise ValueError(f"No promoter sequences loaded from {promoters_path}")
    return promoter_map


def load_utr_block_map(utr5_path: Path) -> UtrBlockMap:
    """Load 5' UTR exon sequences grouped by gene ID and block number."""
    utr_blocks: UtrBlockMap = {}
    for header, sequence in iter_fasta_records(utr5_path):
        gene_id, block_number = parse_utr_header(header)
        utr_blocks.setdefault(gene_id, []).append((block_number, sequence))

    if not utr_blocks:
        raise ValueError(f"No 5' UTR exon sequences loaded from {utr5_path}")
    return utr_blocks


def stitch_utr_blocks(utr_blocks: UtrBlockMap) -> SequenceMap:
    """Concatenate per-gene UTR exon blocks into one contiguous 5' UTR sequence."""
    utr_map: SequenceMap = {}
    for gene_id, blocks in utr_blocks.items():
        sorted_blocks = sorted(blocks, key=lambda item: item[0])
        seen_blocks: set[int] = set()
        sequence_parts: list[str] = []

        for block_number, sequence in sorted_blocks:
            if block_number in seen_blocks:
                raise ValueError(
                    f"Duplicate UTR block {block_number} for gene {gene_id}"
                )
            seen_blocks.add(block_number)
            sequence_parts.append(sequence)

        # Block sorting reconstructs transcript-order 5' UTR from exon FASTA records.
        utr_map[gene_id] = "".join(sequence_parts)

    return utr_map


def load_expression_matrix(rnaseq_path: Path) -> pd.DataFrame:
    """Load the RNA-seq VST matrix with gene IDs in the first column."""
    expression_df = pd.read_csv(rnaseq_path, sep="\t")
    if expression_df.shape[1] == 0:
        raise ValueError(f"RNA-seq matrix has no columns: {rnaseq_path}")

    first_column = expression_df.columns[0]
    expression_df = expression_df.rename(columns={first_column: "gene_id"})
    if expression_df["gene_id"].duplicated().any():
        raise ValueError("RNA-seq matrix contains duplicate gene IDs")
    return expression_df


def extract_tissue_name(sample_column: str) -> str:
    """Extract the tissue prefix from a replicate sample column name."""
    tissue = sample_column.split("_", 1)[0]
    if not tissue:
        raise ValueError(f"Cannot extract tissue from sample column: {sample_column}")
    return tissue


def group_columns_by_tissue(columns: list[str]) -> dict[str, list[str]]:
    """Group replicate sample columns by their tissue prefix."""
    groups: dict[str, list[str]] = {}
    for column in columns:
        tissue = extract_tissue_name(column)
        groups.setdefault(tissue, []).append(column)
    return groups


def average_replicates(expression_df: pd.DataFrame) -> pd.DataFrame:
    """Average replicate columns within each tissue group."""
    sample_columns = [column for column in expression_df.columns if column != "gene_id"]
    if not sample_columns:
        raise ValueError("RNA-seq matrix has no sample columns")

    numeric_df = expression_df[sample_columns].apply(
        pd.to_numeric, errors="raise"
    )
    tissue_groups = group_columns_by_tissue(sample_columns)

    averaged = pd.DataFrame({"gene_id": expression_df["gene_id"]})
    for tissue in sorted(tissue_groups):
        group_columns = tissue_groups[tissue]
        averaged[tissue] = numeric_df[group_columns].mean(axis=1)

    return averaged


def melt_averaged_matrix(averaged_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the averaged tissue matrix into long-form rows."""
    melted = averaged_df.melt(
        id_vars="gene_id",
        var_name="tissue",
        value_name="mean_vst",
    )
    return melted[["gene_id", "tissue", "mean_vst"]]


def assemble_final_rows(
    flat_df: pd.DataFrame,
    promoter_map: SequenceMap,
    utr_map: SequenceMap,
    round_digits: int,
) -> pd.DataFrame:
    """Join expression rows with sequence maps using a strict inner intersection."""
    rows: list[dict[str, object]] = []
    for row in flat_df.itertuples(index=False):
        gene_id = row.gene_id
        tissue = row.tissue
        mean_vst = row.mean_vst

        if gene_id not in promoter_map:
            continue
        if gene_id not in utr_map:
            continue

        rows.append(
            {
                "gene_id": gene_id,
                "tissue": tissue,
                "promoter_sequence": promoter_map[gene_id],
                "utr_5_sequence": utr_map[gene_id],
                "vst_expression": round(float(mean_vst), round_digits),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "gene_id",
            "tissue",
            "promoter_sequence",
            "utr_5_sequence",
            "vst_expression",
        ],
    )


def write_output(output_path: Path, final_df: pd.DataFrame) -> None:
    """Write the final flattened TSV dataset."""
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, sep="\t", index=False)


def parse_arguments() -> argparse.Namespace:
    """Provide documented CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Integrate promoter/5' UTR FASTA sequences with averaged VST expression data."
        )
    )
    parser.add_argument(
        "--rnaseq",
        required=True,
        help="Path to VST expression matrix TSV (genes as rows, replicates as columns).",
    )
    parser.add_argument(
        "--promoters",
        required=True,
        help="Path to promoter FASTA extracted from promoters.bed.",
    )
    parser.add_argument(
        "--utr5-exons",
        required=True,
        help="Path to 5' UTR exon FASTA extracted from utr5_exons.bed.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output flattened TSV dataset.",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=4,
        help="Number of decimal places for vst_expression (default: 4).",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full sequence/expression integration pipeline."""
    rnaseq_path = Path(args.rnaseq)
    promoters_path = Path(args.promoters)
    utr5_path = Path(args.utr5_exons)
    output_path = Path(args.output)

    for input_path in (rnaseq_path, promoters_path, utr5_path):
        if not input_path.exists():
            raise ValueError(f"Input file does not exist: {input_path}")

    if args.round_digits < 0:
        raise ValueError("--round-digits must be non-negative")

    print(f"Loading promoter FASTA: {promoters_path}")
    promoter_map = load_promoter_map(promoters_path)
    print(f"  Promoter genes loaded: {len(promoter_map)}")

    print(f"Loading 5' UTR exon FASTA: {utr5_path}")
    utr_blocks = load_utr_block_map(utr5_path)
    utr_map = stitch_utr_blocks(utr_blocks)
    print(f"  UTR genes loaded: {len(utr_map)}")

    print(f"Loading RNA-seq matrix: {rnaseq_path}")
    expression_df = load_expression_matrix(rnaseq_path)
    averaged_df = average_replicates(expression_df)
    flat_df = melt_averaged_matrix(averaged_df)
    print(f"  Tissues averaged: {averaged_df.shape[1] - 1}")

    final_df = assemble_final_rows(
        flat_df,
        promoter_map,
        utr_map,
        args.round_digits,
    )
    write_output(output_path, final_df)

    print(f"  Final output rows: {len(final_df)}")
    print(f"Saved integrated dataset: {output_path}")


def main() -> None:
    """Convert expected user/data errors into non-zero CLI exits."""
    args = parse_arguments()
    try:
        run_pipeline(args)
    except (ValueError, OSError) as error:
        print(f"Integration error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
