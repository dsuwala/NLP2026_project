#!/usr/bin/env python3
"""
Training loop utilities for the genomic expression Transformer.

Handles device resolution, train/validation DataLoader construction, and
per-epoch MSE (train)/(validation) metric computation.
"""

from __future__ import annotations

import functools

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import collate_expression_batch
from target_normalization import TargetNormalizer


def resolve_device(device_config: str) -> str:
    """Resolve the compute device, falling back to CPU when CUDA is unavailable.

    Args:
        device_config: Requested device string from CONFIG (e.g. ``"cuda"``).

    Returns:
        Resolved device string safe for ``tensor.to(device)``.
    """
    if device_config == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available; using CPU.")
        return "cpu"
    return device_config


def build_dataloaders(
    dataset, config: dict, pad_id: int
) -> tuple[DataLoader, DataLoader, TargetNormalizer]:
    """Split the dataset and build train/validation DataLoaders.

    Fits target normalization on the training split only, then attaches the
    normalizer to the shared dataset so validation uses train statistics.

    Args:
        dataset: Full ``ExpressionDataset`` instance.
        config: Hyperparameter dict with ``val_split`` and ``batch_size``.
        pad_id: Padding token ID passed to the collate function.

    Returns:
        Tuple of (train_loader, val_loader, fitted_target_normalizer).

    Raises:
        ValueError: If the dataset is too small for the requested split.
    """
    val_size = int(len(dataset) * config["val_split"])
    train_size = len(dataset) - val_size

    if val_size < 1 or train_size < 1:
        raise ValueError("Dataset too small for split")

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_raw = np.array(
        [dataset.get_raw_target(i) for i in train_ds.indices], dtype=np.float64
    )
    normalizer = TargetNormalizer()
    normalizer.fit(train_raw)
    dataset.attach_target_normalizer(normalizer)

    collate_fn = functools.partial(collate_expression_batch, pad_id=pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, normalizer


def run_training_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Run one training epoch and return the mean MSE loss.

    Args:
        model: Expression Transformer model.
        train_loader: Training DataLoader yielding (sequences, mask, targets).
        optimizer: Optimizer instance (AdamW).
        criterion: Loss function (MSELoss).
        device: Target device string.

    Returns:
        Mean training MSE loss across all batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for sequences, mask, targets in train_loader:
        sequences = sequences.to(device)
        mask = mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(sequences, mask)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def run_validation_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> float:
    """Run one validation epoch and return the mean squared error (MSE).

    Args:
        model: Expression Transformer model.
        val_loader: Validation DataLoader yielding (sequences, mask, targets).
        device: Target device string.

    Returns:
        Mean squared error in normalized space across all validation samples.
    """
    model.eval()
    total_squared_error = 0.0
    num_samples = 0

    with torch.no_grad():
        for sequences, mask, targets in val_loader:
            sequences = sequences.to(device)
            mask = mask.to(device)
            targets = targets.to(device)

            predictions = model(sequences, mask)
            total_squared_error += torch.sum((predictions - targets) ** 2).item()
            num_samples += targets.numel()

    return total_squared_error / num_samples



def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str,
) -> None:
    """Train the model for the configured number of epochs.

    Logs per-epoch training and validation MSE in normalized target space.

    Args:
        model: Expression Transformer model (already on device).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Hyperparameter dict with ``lr`` and ``epochs``.
        device: Target device string.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()

    for epoch in range(1, config["epochs"] + 1):
        train_mse = run_training_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_mse = run_validation_epoch(model, val_loader, device)
        print(
            f"Epoch {epoch}/{config['epochs']} | "
            f"Train MSE (norm): {train_mse:.6f} | Val MSE (norm): {val_mse:.6f}"
        )
