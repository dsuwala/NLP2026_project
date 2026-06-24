#!/usr/bin/env python3
"""
CLI driver for Drosophila FASTA/GFF synchronization and BED extraction.

Reads a genome FASTA and GFF3 annotation, maps major chromosomes, selects
one transcript per gene by longest 5' UTR, and writes BED6 promoter and
5' UTR exon files.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

ChromosomeMap = dict[str, str]
TranscriptGeneMap = dict[str, str]
UtrBlock = tuple[str, int, int, str, int, str]
TranscriptUtrMap = dict[str, list[UtrBlock]]
SelectedGeneMap = dict[str, tuple[str, list[UtrBlock]]]

MAJOR_CHROMOSOMES = frozenset({"2L", "2R", "3L", "3R", "4", "X", "Y"})


def open_maybe_gzip(path: Path) -> TextIO:
    """Open a plain-text or gzip-compressed file for reading as text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def extract_fasta_headers(fasta_path: Path) -> set[str]:
    """Lazily scan FASTA headers without loading sequence bases."""
    headers: set[str] = set()
    with open_maybe_gzip(fasta_path) as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            header = line[1:].split()[0]
            if header:
                headers.add(header)
    if not headers:
        raise ValueError(f"No FASTA headers found in {fasta_path}")
    return headers


def normalize_chromosome_token(value: str) -> str | None:
    """Convert FASTA/GFF chromosome naming variants into biological chromosome tokens."""
    value = value.strip()
    if not value:
        return None

    token = value.split()[0]

    # FASTA/GFF names are translated through biological chromosome tokens.
    lower_token = token.lower()
    if lower_token.startswith("chr"):
        token = token[3:]
    elif lower_token.startswith("chromosome_"):
        token = token[11:]
    elif lower_token.startswith("chromosome-"):
        token = token[11:]
    elif lower_token.startswith("chromosome"):
        token = token[10:]

    normalized = token.upper()
    if normalized in MAJOR_CHROMOSOMES:
        return normalized
    return None


def build_chromosome_map(fasta_headers: set[str]) -> ChromosomeMap:
    """Map biological chromosomes to exact FASTA header strings."""
    chromosome_map: ChromosomeMap = {}
    for header in sorted(fasta_headers):
        token = normalize_chromosome_token(header)
        if token is None:
            continue
        if token in chromosome_map:
            raise ValueError(
                f"Multiple FASTA headers map to chromosome {token}: "
                f"{chromosome_map[token]} and {header}"
            )
        chromosome_map[token] = header

    missing = sorted(MAJOR_CHROMOSOMES - set(chromosome_map))
    if missing:
        raise ValueError(
            f"FASTA is missing major chromosomes: {', '.join(missing)}"
        )
    return chromosome_map


def parse_gff_attributes(attributes: str) -> dict[str, str]:
    """Parse GFF3 column 9 into a simple key-value dictionary."""
    parsed: dict[str, str] = {}
    for field in attributes.strip().split(";"):
        field = field.strip()
        if not field or "=" not in field:
            continue
        key, value = field.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def first_id(value: str) -> str:
    """Return the first ID when a GFF attribute contains comma-separated values."""
    return value.split(",", 1)[0].strip()


def parse_gff(
    gff_path: Path,
    chromosome_map: ChromosomeMap,
    error_log_path: Path,
) -> tuple[TranscriptGeneMap, TranscriptUtrMap, int]:
    """Read GFF3 once and collect transcript-to-gene relationships plus 5' UTR blocks."""
    transcript_to_gene: TranscriptGeneMap = {}
    transcript_utr_blocks: TranscriptUtrMap = {}
    malformed_count = 0

    with (
        open_maybe_gzip(gff_path) as handle,
        error_log_path.open("w", encoding="utf-8") as error_log,
    ):
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if line == "##FASTA":
                break
            if not line or line.startswith("#"):
                continue

            columns = line.split("\t")
            if len(columns) != 9:
                malformed_count += 1
                error_log.write(
                    f"GFF line {line_number} does not have 9 columns "
                    f"({len(columns)} columns):\t{line}\n"
                )
                continue

            seqid = columns[0]
            feature_type = columns[2]
            start_text = columns[3]
            end_text = columns[4]
            strand = columns[6]
            attributes_text = columns[8]

            token = normalize_chromosome_token(seqid)
            if token is None or token not in chromosome_map:
                continue

            fasta_chrom = chromosome_map[token]

            if feature_type == "mRNA":
                attributes = parse_gff_attributes(attributes_text)
                transcript_id = attributes.get("ID")
                gene_id = attributes.get("Parent")
                if transcript_id is None or gene_id is None:
                    continue
                transcript_to_gene[first_id(transcript_id)] = first_id(gene_id)

            elif feature_type in {"5UTR", "five_prime_UTR"}:
                attributes = parse_gff_attributes(attributes_text)
                parent = attributes.get("Parent")
                if parent is None:
                    continue
                if strand not in {"+", "-"}:
                    malformed_count += 1
                    error_log.write(
                        f"GFF line {line_number} has unsupported UTR strand "
                        f"{strand!r}:\t{line}\n"
                    )
                    continue

                try:
                    start = int(start_text)
                    end = int(end_text)
                except ValueError:
                    malformed_count += 1
                    error_log.write(
                        f"GFF line {line_number} has non-integer coordinates:\t"
                        f"{line}\n"
                    )
                    continue

                if end < start:
                    malformed_count += 1
                    error_log.write(
                        f"GFF line {line_number} has end before start:\t{line}\n"
                    )
                    continue

                parent_transcript = first_id(parent)
                transcript_utr_blocks.setdefault(parent_transcript, []).append(
                    (fasta_chrom, start, end, strand, line_number, line)
                )

    return transcript_to_gene, transcript_utr_blocks, malformed_count


def total_utr_length(blocks: list[UtrBlock]) -> int:
    """Calculate exon-only UTR length for transcript selection."""
    return sum(
        end - start + 1
        for _chrom, start, end, _strand, _line_number, _line in blocks
    )


def select_canonical_transcripts(
    transcript_to_gene: TranscriptGeneMap,
    transcript_utr_blocks: TranscriptUtrMap,
    error_log_path: Path,
) -> tuple[SelectedGeneMap, int]:
    """Pick one transcript per gene using longest total 5' UTR length."""
    # Longest exon-only 5' UTR is used as the deterministic heuristic.
    gene_to_transcripts: dict[str, list[str]] = {}
    for transcript_id, gene_id in transcript_to_gene.items():
        gene_to_transcripts.setdefault(gene_id, []).append(transcript_id)

    selected: SelectedGeneMap = {}
    inconsistent_count = 0
    for gene_id in sorted(gene_to_transcripts):
        candidates: list[tuple[int, str, list[UtrBlock]]] = []
        for transcript_id in sorted(gene_to_transcripts[gene_id]):
            blocks = transcript_utr_blocks.get(transcript_id, [])
            if not blocks:
                continue

            chrom = blocks[0][0]
            strand = blocks[0][3]
            if any(
                block[0] != chrom or block[3] != strand for block in blocks
            ):
                inconsistent_count += len(blocks)
                with error_log_path.open("a", encoding="utf-8") as error_log:
                    for block in blocks:
                        error_log.write(
                            f"Transcript {transcript_id} has inconsistent UTR "
                            "block chromosome or strand; skipped line "
                            f"{block[4]}:\t{block[5]}\n"
                        )
                continue

            candidates.append((total_utr_length(blocks), transcript_id, blocks))

        if not candidates:
            continue

        candidates.sort(key=lambda item: (-item[0], item[1]))
        _length, _transcript_id, blocks = candidates[0]
        selected[gene_id] = (_transcript_id, blocks)

    return selected, inconsistent_count


def sort_blocks_by_coordinate(blocks: list[UtrBlock]) -> list[UtrBlock]:
    """Provide stable genomic sorting for output and TSS logic."""
    return sorted(blocks, key=lambda block: (block[1], block[2], block[0], block[3]))


def calculate_tss(sorted_blocks: list[UtrBlock]) -> int:
    """Find the transcript start site from sorted 5' UTR blocks."""
    if not sorted_blocks:
        raise ValueError("Cannot calculate TSS without UTR blocks")

    strand = sorted_blocks[0][3]
    if strand == "+":
        return sorted_blocks[0][1]
    if strand == "-":
        return sorted_blocks[-1][2]
    raise ValueError(f"Unsupported strand: {strand}")


def calculate_promoter_interval(
    tss: int, strand: str, promoter_length: int
) -> tuple[int, int]:
    """Calculate 1-based inclusive promoter coordinates before BED conversion."""
    if promoter_length < 1:
        raise ValueError("Promoter length must be at least 1")

    if strand == "+":
        start = max(1, tss - promoter_length)
        end = tss - 1
    elif strand == "-":
        start = tss + 1
        end = tss + promoter_length
    else:
        raise ValueError(f"Unsupported strand: {strand}")

    return start, end


def gff_interval_to_bed(start_1based: int, end_1based: int) -> tuple[int, int]:
    """Convert GFF 1-based inclusive intervals into BED 0-based half-open intervals."""
    # GFF3 and BED use different coordinate systems.
    if start_1based < 1:
        raise ValueError("GFF/BED conversion received start < 1")
    if end_1based < start_1based:
        raise ValueError("GFF/BED conversion received end before start")
    return start_1based - 1, end_1based


def format_bed6(
    chrom: str, start_0based: int, end_0based: int, name: str, strand: str
) -> str:
    """Format one BED6 row consistently."""
    return f"{chrom}\t{start_0based}\t{end_0based}\t{name}\t.\t{strand}\n"


def iter_promoter_rows(
    selected: SelectedGeneMap, promoter_length: int
) -> Iterator[str]:
    """Generate all promoter BED6 rows, one per gene."""
    for gene_id in sorted(selected):
        _transcript_id, blocks = selected[gene_id]
        sorted_blocks = sort_blocks_by_coordinate(blocks)
        chrom = sorted_blocks[0][0]
        strand = sorted_blocks[0][3]

        tss = calculate_tss(sorted_blocks)
        promoter_start, promoter_end = calculate_promoter_interval(
            tss, strand, promoter_length
        )
        if promoter_end < promoter_start:
            continue

        bed_start, bed_end = gff_interval_to_bed(promoter_start, promoter_end)
        yield format_bed6(chrom, bed_start, bed_end, gene_id, strand)


def iter_utr5_rows(selected: SelectedGeneMap) -> Iterator[str]:
    """Generate BED6 rows for selected 5' UTR exon blocks."""
    for gene_id in sorted(selected):
        _transcript_id, blocks = selected[gene_id]
        sorted_blocks = sort_blocks_by_coordinate(blocks)
        strand = sorted_blocks[0][3]

        if strand == "+":
            ordered_blocks = sorted_blocks
        elif strand == "-":
            ordered_blocks = list(reversed(sorted_blocks))
        else:
            raise ValueError(f"Unsupported strand: {strand}")

        for block_number, (
            chrom,
            start,
            end,
            block_strand,
            _line_number,
            _line,
        ) in enumerate(
            ordered_blocks, start=1
        ):
            bed_start, bed_end = gff_interval_to_bed(start, end)
            name = f"{gene_id}_block_{block_number}"
            yield format_bed6(chrom, bed_start, bed_end, name, block_strand)


def write_lines(path: Path, rows: Iterator[str]) -> int:
    """Write generated rows and return row count for logging."""
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row)
            count += 1
    return count


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full FASTA/GFF-to-BED pipeline in clear step order."""
    fasta_path = Path(args.fasta)
    gff_path = Path(args.gff)
    output_dir = Path(args.output_dir)

    if not fasta_path.exists():
        raise ValueError(f"FASTA file does not exist: {fasta_path}")
    if not gff_path.exists():
        raise ValueError(f"GFF file does not exist: {gff_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning FASTA headers: {fasta_path}")
    headers = extract_fasta_headers(fasta_path)
    chromosome_map = build_chromosome_map(headers)
    for token in sorted(chromosome_map):
        print(f"  {token} -> {chromosome_map[token]}")

    error_log_path = output_dir / "err.log"
    print(f"Parsing GFF annotations: {gff_path}")
    transcript_to_gene, transcript_utr_blocks, malformed_count = parse_gff(
        gff_path, chromosome_map, error_log_path
    )
    print(f"  Transcripts mapped to genes: {len(transcript_to_gene)}")
    print(f"  Transcripts with 5' UTR blocks: {len(transcript_utr_blocks)}")
    if malformed_count:
        print(
            f"  Skipped malformed GFF lines: {malformed_count} "
            f"(logged to {error_log_path})"
        )

    selected, inconsistent_count = select_canonical_transcripts(
        transcript_to_gene, transcript_utr_blocks, error_log_path
    )
    if inconsistent_count:
        print(
            f"  Skipped inconsistent UTR block lines: {inconsistent_count} "
            f"(logged to {error_log_path})"
        )
    print(f"  Genes with selected canonical transcript: {len(selected)}")

    promoters_path = output_dir / args.promoters_name
    utr5_path = output_dir / args.utr5_name

    promoter_count = write_lines(
        promoters_path, iter_promoter_rows(selected, args.promoter_length)
    )
    utr5_count = write_lines(utr5_path, iter_utr5_rows(selected))

    print(f"Saved promoters: {promoters_path} ({promoter_count} rows)")
    print(f"Saved 5' UTR exons: {utr5_path} ({utr5_count} rows)")


def parse_arguments() -> argparse.Namespace:
    """Provide documented CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate promoter and 5' UTR BED files from Drosophila FASTA/GFF3 inputs."
        )
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to input genome FASTA (.fa or .fa.gz).",
    )
    parser.add_argument(
        "--gff",
        required=True,
        help="Path to input GFF3 annotation (.gff or .gff.gz).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where BED files will be written.",
    )
    parser.add_argument(
        "--promoter-length",
        type=int,
        default=1000,
        help="Promoter window length in base pairs (default: 1000).",
    )
    parser.add_argument(
        "--promoters-name",
        default="promoters.bed",
        help="Output filename for promoter BED rows (default: promoters.bed).",
    )
    parser.add_argument(
        "--utr5-name",
        default="utr5_exons.bed",
        help="Output filename for 5' UTR BED rows (default: utr5_exons.bed).",
    )
    return parser.parse_args()


def main() -> None:
    """Convert expected user/data errors into non-zero CLI exits."""
    args = parse_arguments()
    try:
        run_pipeline(args)
    except (ValueError, OSError) as error:
        print(f"Data translation error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
