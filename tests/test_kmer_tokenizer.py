#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from vocabulary import (  # noqa: E402
    build_vocabulary,
    encode_dna_sequence,
    encoded_dna_length,
    max_raw_dna_length,
)


BASE_CONFIG = {
    "vocab_pad": "[PAD]",
    "vocab_unk": "N",
    "nucleotides": ["A", "C", "T", "G"],
}


class KmerTokenizerTests(unittest.TestCase):
    def test_overlapping_six_mers(self) -> None:
        config = {
            **BASE_CONFIG,
            "tokenization": "kmer",
            "kmer_size": 6,
            "kmer_stride": 1,
        }
        vocabulary = build_vocabulary(["HEAD", "BODY"], config)
        sequence = "ACGATTACAAGT"

        token_ids = encode_dna_sequence(sequence, vocabulary, config)
        decoded = [vocabulary.id_to_token[token_id] for token_id in token_ids]

        self.assertEqual(
            decoded,
            [
                "ACGATT",
                "CGATTA",
                "GATTAC",
                "ATTACA",
                "TTACAA",
                "TACAAG",
                "ACAAGT",
            ],
        )
        self.assertEqual(len(token_ids), encoded_dna_length(len(sequence), config))

    def test_vocabulary_size_for_six_mers(self) -> None:
        config = {
            **BASE_CONFIG,
            "tokenization": "kmer",
            "kmer_size": 6,
            "kmer_stride": 1,
        }
        vocabulary = build_vocabulary(["HEAD", "BODY"], config)
        self.assertEqual(vocabulary.vocab_size, 2 + 4**6 + 2)

    def test_noncanonical_windows_map_to_unk(self) -> None:
        config = {
            **BASE_CONFIG,
            "tokenization": "kmer",
            "kmer_size": 3,
            "kmer_stride": 1,
        }
        vocabulary = build_vocabulary(["HEAD"], config)
        token_ids = encode_dna_sequence("AACNTA", vocabulary, config)
        decoded = [vocabulary.id_to_token[token_id] for token_id in token_ids]
        self.assertEqual(decoded, ["AAC", "N", "N", "N"])

    def test_nucleotide_mode_is_backward_compatible(self) -> None:
        config = {**BASE_CONFIG, "tokenization": "nucleotide"}
        vocabulary = build_vocabulary(["HEAD"], config)
        token_ids = encode_dna_sequence("ACTGN", vocabulary, config)
        decoded = [vocabulary.id_to_token[token_id] for token_id in token_ids]
        self.assertEqual(decoded, ["A", "C", "T", "G", "N"])

    def test_short_sequence_produces_no_complete_kmer(self) -> None:
        config = {
            **BASE_CONFIG,
            "tokenization": "kmer",
            "kmer_size": 6,
            "kmer_stride": 1,
        }
        vocabulary = build_vocabulary(["HEAD"], config)
        self.assertEqual(encode_dna_sequence("ACGT", vocabulary, config), [])
        self.assertEqual(encoded_dna_length(4, config), 0)

    def test_raw_length_and_token_budget_are_consistent(self) -> None:
        config = {
            **BASE_CONFIG,
            "tokenization": "kmer",
            "kmer_size": 6,
            "kmer_stride": 1,
        }
        max_dna_tokens = 95
        raw_length = max_raw_dna_length(max_dna_tokens, config)
        self.assertEqual(raw_length, 100)
        self.assertEqual(encoded_dna_length(raw_length, config), max_dna_tokens)

    def test_2200_nt_requires_2196_total_tokens(self) -> None:
        config = {
            **BASE_CONFIG,
            "tokenization": "kmer",
            "kmer_size": 6,
            "kmer_stride": 1,
        }
        self.assertEqual(encoded_dna_length(2200, config), 2195)
        self.assertEqual(encoded_dna_length(2200, config) + 1, 2196)


if __name__ == "__main__":
    unittest.main()
