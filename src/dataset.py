#!/usr/bin/env python3
"""
PyTorch Dataset and batch collation for genomic expression regression.

Assembles promoter + 5' UTR DNA sequences and tissue labels separately for
CNN/max-pool preprocessing before tissue is prepended inside the model.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from target_normalization import TargetNormalizer
from vocabulary import Vocabulary, encode_dna_sequence, encode_tissue


class ExpressionDataset(Dataset):
    """Dataset mapping (tissue, DNA sequence) pairs to VST expression targets.

    Each sample returns a tissue token ID, a DNA-only token tensor, and a
    scalar regression target. Tissue is prepended after CNN pooling in the model.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocabulary: Vocabulary,
        config: dict,
        sorted_tissues: list[str],
        target_normalizer: TargetNormalizer | None = None,
    ) -> None:
        """Initialize the dataset with pre-loaded data and vocabulary.

        Args:
            dataframe: Expression DataFrame with tissue, sequence, and target columns.
            vocabulary: Token vocabulary built from the dataset tissues.
            config: Hyperparameter dict (uses ``max_seq_len`` for DNA only).
            sorted_tissues: Alphabetically sorted tissue names for encoding.
            target_normalizer: Optional fitted scaler for regression targets.
        """
        self._df = dataframe.reset_index(drop=True)
        self._vocabulary = vocabulary
        self._config = config
        self._sorted_tissues = sorted_tissues
        self._max_seq_len = config["max_seq_len"]
        self._target_normalizer = target_normalizer

    def attach_target_normalizer(self, normalizer: TargetNormalizer) -> None:
        """Attach a train-fitted normalizer for target transformation.

        Args:
            normalizer: Fitted ``TargetNormalizer`` using training-split stats.
        """
        self._target_normalizer = normalizer

    def get_raw_target(self, index: int) -> float:
        """Return the unnormalized VST expression for a dataset row.

        Args:
            index: Row index into the underlying DataFrame.

        Returns:
            Raw ``vst_expression`` value before normalization.
        """
        return float(self._df.iloc[index]["vst_expression"])

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch one tokenized sample and its VST expression target.

        Args:
            index: Row index into the underlying DataFrame.

        Returns:
            Tuple of (tissue_id, dna_tokens, target) where tissue_id is a
            scalar long tensor, dna_tokens has shape (L,), and target is a
            scalar float tensor. DNA longer than ``max_seq_len`` is truncated.
        """
        row = self._df.iloc[index]
        tissue = str(row["tissue"])
        promoter = str(row["promoter_sequence"])
        utr = str(row["utr_5_sequence"])

        # Promoter + UTR_5 per spec
        combined = promoter + utr
        dna_ids = encode_dna_sequence(combined, self._vocabulary, self._config)
        tissue_id = encode_tissue(tissue, self._vocabulary, self._sorted_tissues)

        if len(dna_ids) > self._max_seq_len:
            dna_ids = dna_ids[: self._max_seq_len]

        raw_target = float(row["vst_expression"])
        if self._target_normalizer is not None:
            # Normalized target (min-max + z-score) when normalizer attached
            raw_target = float(self._target_normalizer.transform(raw_target))
        target = torch.tensor(raw_target, dtype=torch.float32)
        return (
            torch.tensor(tissue_id, dtype=torch.long),
            torch.tensor(dna_ids, dtype=torch.long),
            target,
        )


def collate_expression_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad DNA sequences and build batch tensors for training.

    Args:
        batch: List of (tissue_id, dna_tokens, target) tuples from the Dataset.
        pad_id: Integer ID of the padding token.

    Returns:
        Tuple of (tissue_ids, dna_tokens, dna_padding_mask, targets) with shapes
        (B,), (B, L_max), (B, L_max), and (B,) respectively.
    """
    tissue_ids = torch.stack([item[0] for item in batch], dim=0)
    dna_list = [item[1] for item in batch]
    targets = torch.stack([item[2] for item in batch], dim=0)

    # Shape: (B, L_max)
    dna_padded = pad_sequence(dna_list, batch_first=True, padding_value=pad_id)

    # True = ignore (padding); Shape: (B, L_max)
    dna_padding_mask = dna_padded.eq(pad_id)

    return tissue_ids, dna_padded, dna_padding_mask, targets
