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


def get_lr_for_epoch(epoch: int, config: dict) -> float:
    """Compute the learning rate for a 1-indexed training epoch.

    Linearly ramps from ``start_lr`` at epoch 1 to ``max_lr`` at epoch
    ``warmup_period``, then holds ``max_lr`` constant.

    Args:
        epoch: Current training epoch (1-indexed).
        config: Hyperparameter dict with ``start_lr``, ``max_lr``, and
            ``warmup_period``.

    Returns:
        Learning rate to use for the given epoch.
    """
    start_lr = config["start_lr"]
    max_lr = config["max_lr"]
    warmup_period = config["warmup_period"]

    if warmup_period <= 1:
        return max_lr
    if epoch >= warmup_period:
        return max_lr

    progress = (epoch - 1) / (warmup_period - 1)
    return start_lr + progress * (max_lr - start_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set the learning rate for all optimizer parameter groups.

    Args:
        optimizer: Optimizer whose param groups will be updated.
        lr: New learning rate value.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def run_training_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    gradient_accumulation_steps: int,
) -> float:
    """Run one training epoch and return the mean MSE loss.

    Args:
        model: Expression Transformer model.
        train_loader: Training DataLoader yielding
            (tissue_ids, dna_tokens, dna_mask, targets).
        optimizer: Optimizer instance (AdamW).
        criterion: Loss function (MSELoss).
        device: Target device string.
        gradient_accumulation_steps: Micro-batches to accumulate before stepping.

    Returns:
        Mean training MSE loss across all batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    accum_steps = gradient_accumulation_steps
    num_train_batches = len(train_loader)

    optimizer.zero_grad()

    for batch_idx, (tissue_ids, dna_tokens, dna_mask, targets) in enumerate(
        train_loader
    ):
        tissue_ids = tissue_ids.to(device)
        dna_tokens = dna_tokens.to(device)
        dna_mask = dna_mask.to(device)
        targets = targets.to(device)

        predictions = model(tissue_ids, dna_tokens, dna_mask)
        loss = criterion(predictions, targets)
        scaled_loss = loss / accum_steps
        scaled_loss.backward()

        total_loss += loss.item()
        num_batches += 1

        is_accum_step = (batch_idx + 1) % accum_steps == 0
        is_last_batch = (batch_idx + 1) == num_train_batches

        if is_accum_step or is_last_batch:
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / num_batches


def run_validation_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> float:
    """Run one validation epoch and return the mean squared error (MSE).

    Args:
        model: Expression Transformer model.
        val_loader: Validation DataLoader yielding
            (tissue_ids, dna_tokens, dna_mask, targets).
        device: Target device string.

    Returns:
        Mean squared error in normalized space across all validation samples.
    """
    model.eval()
    total_squared_error = 0.0
    num_samples = 0

    with torch.no_grad():
        for tissue_ids, dna_tokens, dna_mask, targets in val_loader:
            tissue_ids = tissue_ids.to(device)
            dna_tokens = dna_tokens.to(device)
            dna_mask = dna_mask.to(device)
            targets = targets.to(device)

            predictions = model(tissue_ids, dna_tokens, dna_mask)
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

    Logs per-epoch learning rate, training MSE, and validation MSE in normalized
    target space. Effective batch size is
    ``batch_size * gradient_accumulation_steps``.

    Args:
        model: Expression Transformer model (already on device).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Hyperparameter dict with ``start_lr``, ``max_lr``,
            ``warmup_period``, ``gradient_accumulation_steps``, and ``epochs``.
        device: Target device string.
    """
    accum_steps = config["gradient_accumulation_steps"]
    if accum_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["start_lr"])
    criterion = nn.MSELoss()

    for epoch in range(1, config["epochs"] + 1):
        current_lr = get_lr_for_epoch(epoch, config)
        set_optimizer_lr(optimizer, current_lr)

        train_mse = run_training_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            accum_steps,
        )
        val_mse = run_validation_epoch(model, val_loader, device)
        print(
            f"Epoch {epoch}/{config['epochs']} | LR: {current_lr:.2e} | "
            f"Train MSE (norm): {train_mse:.6f} | Val MSE (norm): {val_mse:.6f}"
        )
