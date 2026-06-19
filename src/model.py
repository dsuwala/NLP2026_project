#!/usr/bin/env python3
"""
Minimalist Pre-LayerNorm Transformer for genomic expression regression.

DNA tokens are embedded, processed by a 1D CNN and max-pool, then a tissue
token is prepended before the Transformer stack. ``max_seq_len`` in CONFIG is
DNA-only; the tissue token is added in the model after local preprocessing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1d_output_length(
    seq_len: int, kernel: int, stride: int, padding: int
) -> int:
    """Compute Conv1d output length for a 1-D sequence.

    Args:
        seq_len: Input sequence length.
        kernel: Convolution kernel size.
        stride: Convolution stride.
        padding: Zero-padding on both sides.

    Returns:
        Output sequence length after convolution.
    """
    return (seq_len + 2 * padding - kernel) // stride + 1


def maxpool1d_output_length(seq_len: int, kernel: int, stride: int) -> int:
    """Compute MaxPool1d output length for a 1-D sequence.

    Args:
        seq_len: Input sequence length.
        kernel: Pooling kernel size.
        stride: Pooling stride.

    Returns:
        Output sequence length after max pooling.
    """
    return (seq_len - kernel) // stride + 1


def _align_mask_to_length(mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Crop or pad a boolean padding mask to match a sequence length.

    Args:
        mask: Boolean mask of shape (B, L_mask); True marks padding.
        seq_len: Target sequence length L_target.

    Returns:
        Mask of shape (B, L_target).
    """
    if mask.shape[1] == seq_len:
        return mask
    if mask.shape[1] < seq_len:
        return F.pad(mask, (0, seq_len - mask.shape[1]), value=True)
    return mask[:, :seq_len]


def _pad_sequence_to_length(
    x: torch.Tensor, mask: torch.Tensor, target_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad or crop sequence tensors to a fixed length for the Transformer.

    Args:
        x: Feature tensor of shape (B, L, D).
        mask: Boolean padding mask of shape (B, L); True marks padding.
        target_len: Target sequence length.

    Returns:
        Tuple of (x, mask) both with sequence length ``target_len``.
    """
    mask = _align_mask_to_length(mask, x.shape[1])
    current_len = x.shape[1]
    if current_len == target_len:
        return x, mask
    if current_len > target_len:
        return x[:, :target_len], _align_mask_to_length(mask, target_len)
    pad_len = target_len - current_len
    x = F.pad(x, (0, 0, 0, pad_len))
    mask = F.pad(mask, (0, pad_len), value=True)
    return x, mask


class TransformerBlock(nn.Module):
    """Single Pre-LN transformer block with multi-head self-attention and MLP.

    Residual connections wrap both the attention and feed-forward sub-layers.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_mlp: int, dropout: float
    ) -> None:
        """Initialize layer norms, attention, and feed-forward MLP.

        Args:
            d_model: Model embedding dimension.
            n_heads: Number of attention heads.
            d_mlp: Hidden dimension of the feed-forward network.
            dropout: Dropout probability applied in attention and MLP.
        """
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply Pre-LN self-attention and MLP with residual connections.

        Args:
            x: Input tensor of shape (B, L, D).
            key_padding_mask: Boolean mask of shape (B, L); True marks padding.

        Returns:
            Output tensor of shape (B, L, D).
        """
        norm_x = self.layer_norm1(x)
        attn_out, _ = self.attention(
            norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask
        )
        # Residual after attention
        x = x + attn_out

        norm_x2 = self.layer_norm2(x)
        mlp_out = self.mlp(norm_x2)
        # Residual after MLP
        x = x + mlp_out

        # Shape: (B, L, D)
        return x


class Conv1dBlock(nn.Module):
    """Local 1D convolution over the DNA sequence before max pooling.

    Applies Conv1d with GELU and dropout. Padded positions are zeroed on input
    and the padding mask is propagated to match the conv output length.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dropout: float,
    ) -> None:
        """Initialize the 1D convolution block.

        Args:
            d_model: Channel dimension (input and output).
            kernel_size: Convolution kernel width along the sequence axis.
            stride: Convolution stride along the sequence axis.
            padding: Zero-padding applied to both sides of the sequence axis.
            dropout: Dropout probability after activation.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply local convolution over the DNA sequence.

        Args:
            x: Input tensor of shape (B, L, D).
            padding_mask: Boolean mask of shape (B, L); True marks padding.

        Returns:
            Tuple of (output, output_mask) with shapes (B, L_out, D) and
            (B, L_out) respectively.
        """
        padding_mask = _align_mask_to_length(padding_mask, x.shape[1])
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Conv1d expects (B, D, L)
        x = x.transpose(1, 2)
        x = self.conv(x)
        l_out = x.shape[2]
        x = x.transpose(1, 2)

        x = self.dropout(self.activation(x))

        # Propagate validity mask through the conv receptive field
        valid = (~padding_mask).to(dtype=x.dtype).unsqueeze(1)
        valid_weight = torch.ones(
            1, 1, self.kernel_size, device=x.device, dtype=x.dtype
        )
        valid_out = F.conv1d(
            valid,
            valid_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )
        output_mask = valid_out.squeeze(1) == 0
        output_mask = _align_mask_to_length(output_mask, l_out)

        # Shape: (B, L_out, D)
        return x, output_mask


class MaxPool1dBlock(nn.Module):
    """Max pooling over the convolved DNA sequence dimension."""

    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        """Initialize the max-pooling block.

        Args:
            kernel_size: Pooling kernel width along the sequence axis.
            stride: Pooling stride; defaults to ``kernel_size``.
        """
        super().__init__()
        pool_stride = stride if stride is not None else kernel_size
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=pool_stride)

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply max pooling and propagate the padding mask.

        Args:
            x: Input tensor of shape (B, L, D).
            padding_mask: Boolean mask of shape (B, L); True marks padding.

        Returns:
            Tuple of (pooled_x, pooled_mask) with shapes (B, L_pool, D) and
            (B, L_pool) respectively.
        """
        padding_mask = _align_mask_to_length(padding_mask, x.shape[1])

        x = x.transpose(1, 2)
        pooled = self.pool(x)

        valid = (~padding_mask).float().unsqueeze(1)
        pooled_valid = self.pool(valid)
        pooled_mask = pooled_valid.squeeze(1) == 0
        pooled_mask = _align_mask_to_length(pooled_mask, pooled.shape[2])

        pooled = pooled.transpose(1, 2)
        # Shape: (B, L_pool, D)
        return pooled, pooled_mask


class ExpressionTransformer(nn.Module):
    """Transformer encoder that regresses VST expression from genomic tokens.

    DNA passes through CNN and max-pool; a tissue embedding is prepended before
    the Transformer. The tissue token at index 0 is routed to the regression head.
    """

    def __init__(self, vocab_size: int, pad_id: int, config: dict) -> None:
        """Build embeddings, CNN/pool stack, transformer, and regression head.

        Args:
            vocab_size: Total number of tokens in the vocabulary.
            pad_id: Padding token index for the embedding layer.
            config: Hyperparameter dict with model architecture keys.
        """
        super().__init__()
        d_model = config["d_model"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        d_mlp = config["d_mlp"]
        dropout = config["dropout"]
        max_dna_len = config["max_seq_len"]
        kernel = config["cnn_kernel_size"]
        stride = config["cnn_stride"]
        padding = config["cnn_padding"]
        pool_size = config["cnn_max_pool_size"]

        if kernel < 1 or stride < 1 or padding < 0:
            raise ValueError("Invalid CNN hyperparameters.")
        if pool_size < 1:
            raise ValueError("cnn_max_pool_size must be >= 1.")

        l_conv = conv1d_output_length(max_dna_len, kernel, stride, padding)
        l_pool = maxpool1d_output_length(l_conv, pool_size, pool_size)
        transformer_max_len = l_pool + 1

        self._max_dna_len = max_dna_len
        self._pooled_max_len = l_pool
        self._transformer_max_len = transformer_max_len

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_id
        )
        self.dna_positional_embedding = nn.Embedding(max_dna_len, d_model)
        self.transformer_positional_embedding = nn.Embedding(
            transformer_max_len, d_model
        )
        self.conv_block = Conv1dBlock(d_model, kernel, stride, padding, dropout)
        self.pool_block = MaxPool1dBlock(pool_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_mlp, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        tissue_ids: torch.Tensor,
        dna_tokens: torch.Tensor,
        dna_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode DNA and tissue tokens and predict scalar VST expression.

        Args:
            tissue_ids: Tissue token IDs of shape (B,).
            dna_tokens: DNA token IDs of shape (B, L_dna).
            dna_padding_mask: Boolean DNA padding mask of shape (B, L_dna).

        Returns:
            Predicted expression values of shape (B,).
        """
        batch_size, dna_len = dna_tokens.shape
        if dna_len > self._max_dna_len:
            dna_tokens = dna_tokens[:, : self._max_dna_len]
            dna_padding_mask = dna_padding_mask[:, : self._max_dna_len]
            dna_len = self._max_dna_len
        dna_padding_mask = _align_mask_to_length(dna_padding_mask, dna_len)

        # DNA embeddings — Shape: (B, L_dna, D)
        dna_x = self.token_embedding(dna_tokens)
        dna_positions = torch.arange(dna_len, device=dna_tokens.device).unsqueeze(
            0
        ).expand(batch_size, dna_len)
        dna_x = dna_x + self.dna_positional_embedding(dna_positions)

        # CNN local features — Shape: (B, L_conv, D)
        dna_x, conv_mask = self.conv_block(dna_x, dna_padding_mask)

        # Max pool — Shape: (B, L_pool, D), variable per batch
        pooled_x, pooled_mask = self.pool_block(dna_x, conv_mask)

        # Pad to fixed length so Transformer input matches positional embeddings
        pooled_x, pooled_mask = _pad_sequence_to_length(
            pooled_x, pooled_mask, self._pooled_max_len
        )

        # Tissue embedding at transformer position 0 — Shape: (B, 1, D)
        tissue_emb = self.token_embedding(tissue_ids).unsqueeze(1)
        tissue_emb = tissue_emb + self.transformer_positional_embedding(
            torch.zeros(batch_size, dtype=torch.long, device=tissue_ids.device)
        )

        # Positions 1..pooled_max_len for pooled DNA (padded slots stay masked)
        pool_positions = torch.arange(
            1, self._pooled_max_len + 1, device=dna_tokens.device
        ).unsqueeze(0).expand(batch_size, self._pooled_max_len)
        pooled_x = pooled_x + self.transformer_positional_embedding(pool_positions)

        # Prepend tissue — Shape: (B, pooled_max_len+1, D)
        x = torch.cat([tissue_emb, pooled_x], dim=1)
        tissue_not_masked = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=dna_tokens.device
        )
        key_padding_mask = torch.cat([tissue_not_masked, pooled_mask], dim=1)
        key_padding_mask = _align_mask_to_length(key_padding_mask, x.shape[1])

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, key_padding_mask)

        x = self.final_layer_norm(x)

        # Route tissue token at index 0 — Shape: (B, D)
        cls_token = x[:, 0, :]

        # Regression head — Shape: (B,) after squeeze
        out = self.regression_head(cls_token).squeeze(-1)
        return out
