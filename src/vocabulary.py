#!/usr/bin/env python3

"""
Integer vocabulary for the genomic expression Transformer.

Supports two DNA tokenization modes:

* ``nucleotide``: one token per nucleotide (the original modelV1 baseline),
* ``kmer``: overlapping canonical k-mers, DNABERT-style.

Special tokens, DNA tokens and dataset-specific tissue tokens share one
embedding table.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass
class Vocabulary:
    """Bidirectional token-to-ID mapping with padding metadata."""

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    pad_id: int
    vocab_size: int


def build_tissue_token_names(sorted_tissues: list[str]) -> list[str]:
    """Build tissue placeholder names in 1-based index order."""

    return [f"[TISSUE_{i}]" for i in range(1, len(sorted_tissues) + 1)]


def get_tokenization_mode(config: dict) -> str:
    """Return and validate the configured DNA tokenization mode."""

    mode = str(config.get("tokenization", "nucleotide")).lower()
    if mode not in {"nucleotide", "kmer"}:
        raise ValueError(
            "config['tokenization'] must be either 'nucleotide' or 'kmer', "
            f"got {mode!r}."
        )
    return mode


def get_kmer_parameters(config: dict) -> tuple[int, int]:
    """Return validated ``(kmer_size, kmer_stride)`` values."""

    kmer_size = int(config.get("kmer_size", 6))
    kmer_stride = int(config.get("kmer_stride", 1))

    if kmer_size <= 0:
        raise ValueError("config['kmer_size'] must be a positive integer.")
    if kmer_stride <= 0:
        raise ValueError("config['kmer_stride'] must be a positive integer.")

    return kmer_size, kmer_stride


def build_kmer_tokens(nucleotides: list[str], kmer_size: int) -> list[str]:
    """Enumerate all canonical k-mers in deterministic order."""

    if kmer_size <= 0:
        raise ValueError("kmer_size must be positive.")
    if not nucleotides:
        raise ValueError("nucleotides cannot be empty.")

    normalized = [str(base).upper() for base in nucleotides]
    if len(set(normalized)) != len(normalized):
        raise ValueError("nucleotides must not contain duplicates.")
    if any(len(base) != 1 for base in normalized):
        raise ValueError("Every nucleotide symbol must contain one character.")

    return ["".join(chars) for chars in product(normalized, repeat=kmer_size)]


def build_vocabulary(unique_tissues: list[str], config: dict) -> Vocabulary:
    """Construct the vocabulary for the selected tokenization mode.

    Token order is deterministic: ``PAD, UNK, DNA tokens, tissue tokens``.
    In k-mer mode, windows containing a noncanonical character are represented
    by the shared UNK token and are not added separately to the vocabulary.
    """

    pad_token = str(config["vocab_pad"])
    unk_token = str(config["vocab_unk"])
    nucleotides = [str(base).upper() for base in config["nucleotides"]]
    mode = get_tokenization_mode(config)

    if pad_token == unk_token:
        raise ValueError("PAD and UNK tokens must be different.")

    if mode == "nucleotide":
        dna_tokens = nucleotides
    else:
        kmer_size, _ = get_kmer_parameters(config)
        dna_tokens = build_kmer_tokens(nucleotides, kmer_size)

    tissue_tokens = build_tissue_token_names(sorted(unique_tissues))
    all_tokens = [pad_token, unk_token, *dna_tokens, *tissue_tokens]

    if len(set(all_tokens)) != len(all_tokens):
        raise ValueError(
            "Vocabulary tokens are not unique. Check PAD/UNK names, "
            "nucleotide symbols and tissue token construction."
        )

    token_to_id = {token: index for index, token in enumerate(all_tokens)}
    id_to_token = {index: token for token, index in token_to_id.items()}

    return Vocabulary(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_id=token_to_id[pad_token],
        vocab_size=len(all_tokens),
    )


def _encode_nucleotides(
    sequence: str,
    vocabulary: Vocabulary,
    config: dict,
) -> list[int]:
    """Encode one token per nucleotide."""

    unk_id = vocabulary.token_to_id[str(config["vocab_unk"])]
    nucleotide_set = {str(base).upper() for base in config["nucleotides"]}

    return [
        vocabulary.token_to_id.get(base, unk_id)
        if base in nucleotide_set
        else unk_id
        for base in sequence.upper()
    ]


def _encode_kmers(
    sequence: str,
    vocabulary: Vocabulary,
    config: dict,
) -> list[int]:
    """Encode overlapping k-mers using the configured stride.

    DNABERT-like tokenization corresponds to ``kmer_size=6`` and
    ``kmer_stride=1``. A window containing ``N`` or another noncanonical
    character maps to the shared UNK token.
    """

    kmer_size, kmer_stride = get_kmer_parameters(config)
    sequence = sequence.upper()
    unk_id = vocabulary.token_to_id[str(config["vocab_unk"])]

    if len(sequence) < kmer_size:
        return []

    return [
        vocabulary.token_to_id.get(sequence[start : start + kmer_size], unk_id)
        for start in range(0, len(sequence) - kmer_size + 1, kmer_stride)
    ]


def encode_dna_sequence(
    sequence: str,
    vocabulary: Vocabulary,
    config: dict,
) -> list[int]:
    """Convert DNA into nucleotide or overlapping k-mer token IDs."""

    if get_tokenization_mode(config) == "nucleotide":
        return _encode_nucleotides(sequence, vocabulary, config)
    return _encode_kmers(sequence, vocabulary, config)


def encoded_dna_length(sequence_length: int, config: dict) -> int:
    """Return the number of DNA tokens produced from a raw sequence length.

    The prepended tissue token is not included.
    """

    if sequence_length < 0:
        raise ValueError("sequence_length cannot be negative.")

    if get_tokenization_mode(config) == "nucleotide":
        return sequence_length

    kmer_size, kmer_stride = get_kmer_parameters(config)
    if sequence_length < kmer_size:
        return 0
    return 1 + (sequence_length - kmer_size) // kmer_stride


def max_raw_dna_length(max_dna_tokens: int, config: dict) -> int:
    """Return the largest raw DNA prefix fitting in ``max_dna_tokens``.

    For overlapping k-mers this converts a token budget back to a nucleotide
    budget. The returned length produces at most ``max_dna_tokens`` DNA tokens.
    """

    if max_dna_tokens < 0:
        raise ValueError("max_dna_tokens cannot be negative.")

    if get_tokenization_mode(config) == "nucleotide":
        return max_dna_tokens

    if max_dna_tokens == 0:
        return 0

    kmer_size, kmer_stride = get_kmer_parameters(config)
    return kmer_size + (max_dna_tokens - 1) * kmer_stride


def encode_tissue(
    tissue_name: str,
    vocabulary: Vocabulary,
    sorted_tissues: list[str],
) -> int:
    """Map a tissue name to its dedicated ``[TISSUE_i]`` token ID."""

    try:
        tissue_index = sorted_tissues.index(tissue_name)
    except ValueError as exc:
        raise ValueError(f"Unknown tissue name: {tissue_name}") from exc

    return vocabulary.token_to_id[f"[TISSUE_{tissue_index + 1}]"]
