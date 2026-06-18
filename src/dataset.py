#!/usr/bin/env python3
"""
PyTorch Dataset and batch collation for genomic expression regression.

Assembles promoter + 5' UTR sequences, prepends a tissue token, and provides
dynamic padding with attention masks for variable-length batches.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from target_normalization import TargetNormalizer
from vocabulary import Vocabulary, encode_dna_sequence, encode_tissue


class ExpressionDataset(Dataset):
    """Dataset mapping (tissue + DNA sequence) pairs to VST expression targets.

    Each sample returns a 1-D token tensor with the tissue token prepended at
    index 0, plus a scalar regression target.
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
            config: Hyperparameter dict (uses ``max_seq_len``).
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch one tokenized sample and its VST expression target.

        Args:
            index: Row index into the underlying DataFrame.

        Returns:
            Tuple of (token_ids, target) where token_ids has shape (L,) and
            target is a scalar float tensor.

        Raises:
            ValueError: If the assembled token sequence exceeds ``max_seq_len``.
        """
        row = self._df.iloc[index]
        tissue = str(row["tissue"])
        promoter = str(row["promoter_sequence"])
        utr = str(row["utr_5_sequence"])

        # Promoter + UTR_5 per spec
        combined = promoter + utr
        dna_ids = encode_dna_sequence(combined, self._vocabulary, self._config)
        tissue_id = encode_tissue(tissue, self._vocabulary, self._sorted_tissues)

        # Prepend tissue token at index 0
        token_ids = [tissue_id] + dna_ids

        if len(token_ids) > self._max_seq_len:
            raise ValueError(
                f"Sample {index} length {len(token_ids)} exceeds "
                f"max_seq_len {self._max_seq_len}"
            )

        raw_target = float(row["vst_expression"])
        if self._target_normalizer is not None:
            # Normalized target (min-max + z-score) when normalizer attached
            raw_target = float(self._target_normalizer.transform(raw_target))
        target = torch.tensor(raw_target, dtype=torch.float32)
        tokens = torch.tensor(token_ids, dtype=torch.long)
        return tokens, target


def collate_expression_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences and build an attention padding mask.

    Args:
        batch: List of (token_tensor, target_scalar) tuples from the Dataset.
        pad_id: Integer ID of the padding token.

    Returns:
        Tuple of (sequences, key_padding_mask, targets) with shapes
        (B, L_max), (B, L_max), and (B,) respectively.
    """
    sequences_list, targets_list = zip(*batch)

    # Shape: (B, L_max)
    padded = pad_sequence(sequences_list, batch_first=True, padding_value=pad_id)

    # Shape: (B,)
    targets = torch.stack(targets_list, dim=0)

    # True = ignore (padding); Shape: (B, L_max)
    key_padding_mask = padded.eq(pad_id)

    return padded, key_padding_mask, targets
