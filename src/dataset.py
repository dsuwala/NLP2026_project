#!/usr/bin/env python3
"""Dataset and collation for the no-CNN genomic Transformer.

The tissue token is prepended to the nucleotide sequence inside the dataset.
``max_seq_len`` therefore denotes the total Transformer input length, including
one tissue token. DNA exceeding ``max_seq_len - 1`` is truncated deterministically
from the right.
"""
from __future__ import annotations

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from target_normalization import TargetNormalizer
from vocabulary import Vocabulary, encode_dna_sequence, encode_tissue


class ExpressionDataset(Dataset):
    """Map a tissue and promoter/5' UTR sequence to an expression target.

    Each sample returns ``(tokens, target)``. ``tokens[0]`` is the tissue token;
    the remaining positions contain promoter followed by 5' UTR nucleotides.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocabulary: Vocabulary,
        config: dict,
        sorted_tissues: list[str],
        target_normalizer: TargetNormalizer | None = None,
    ) -> None:
        self._df = dataframe.reset_index(drop=True)
        self._vocabulary = vocabulary
        self._config = config
        self._sorted_tissues = list(sorted_tissues)
        self._max_seq_len = int(config["max_seq_len"])
        self._target_normalizer = target_normalizer

        if self._max_seq_len < 2:
            raise ValueError(
                "max_seq_len must be at least 2: one tissue token and at "
                "least one DNA token."
            )

    @property
    def max_seq_len(self) -> int:
        """Maximum total token count, including the tissue token."""

        return self._max_seq_len

    @property
    def max_visible_dna_len(self) -> int:
        """Maximum number of DNA characters visible to the model."""

        return self._max_seq_len - 1

    def attach_target_normalizer(self, normalizer: TargetNormalizer) -> None:
        """Attach a normalizer fitted only on the training split."""

        self._target_normalizer = normalizer

    def get_raw_target(self, index: int) -> float:
        """Return the unnormalized VST expression value for one row."""

        return float(self._df.iloc[index]["vst_expression"])

    def get_full_dna_sequence(self, index: int) -> str:
        """Return promoter + 5' UTR before truncation, in uppercase."""

        row = self._df.iloc[index]
        promoter = str(row["promoter_sequence"])
        utr = str(row["utr_5_sequence"])
        return (promoter + utr).upper()

    def get_visible_dna_sequence(self, index: int) -> str:
        """Return exactly the DNA prefix that can be presented to the model."""

        return self.get_full_dna_sequence(index)[: self.max_visible_dna_len]

    def is_truncated(self, index: int) -> bool:
        """Report whether the row's DNA exceeds the visible input window."""

        return len(self.get_full_dna_sequence(index)) > self.max_visible_dna_len

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(tokens, target)`` for the no-CNN model interface.

        The returned sequence is always at most ``max_seq_len`` tokens long.
        Truncation is applied to DNA only, preserving the tissue token at index 0.
        """

        row = self._df.iloc[index]
        tissue = str(row["tissue"])
        visible_dna = self.get_visible_dna_sequence(index)

        dna_ids = encode_dna_sequence(
            visible_dna,
            self._vocabulary,
            self._config,
        )
        tissue_id = encode_tissue(
            tissue,
            self._vocabulary,
            self._sorted_tissues,
        )
        token_ids = [tissue_id, *dna_ids]

        raw_target = float(row["vst_expression"])
        if self._target_normalizer is not None:
            raw_target = float(self._target_normalizer.transform(raw_target))

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(raw_target, dtype=torch.float32),
        )


def collate_expression_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length token sequences and build a padding mask.

    Returns:
        ``(tokens, key_padding_mask, targets)`` where ``True`` in the mask marks
        padding positions ignored by Transformer attention.
    """

    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    sequences, targets = zip(*batch)
    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=pad_id,
    )
    key_padding_mask = padded.eq(pad_id)
    return padded, key_padding_mask, torch.stack(targets, dim=0)
