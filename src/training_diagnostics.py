#!/usr/bin/env python3
"""Training utilities with four-group diagnostics and optional balancing."""
from __future__ import annotations

import functools
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset import collate_expression_batch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from callbacks import BestCheckpointEarlyStopping
from diagnostics import (
    GROUP_ORDER,
    InteractionMetrics,
    RegressionMetrics,
    build_group_labels,
    compute_group_metrics,
    compute_interaction_metrics,
    compute_regression_metrics,
    count_groups,
    display_group_name,
    format_group_metrics,
)
from training import get_lr_for_epoch, set_optimizer_lr
from target_normalization import TargetNormalizer


@dataclass(frozen=True)
class DiagnosticDataBundle:
    """DataLoaders, split metadata, and the train-fitted normalizer."""
    train_loader: DataLoader
    val_loader: DataLoader
    normalizer: TargetNormalizer
    train_indices: list[int]
    val_indices: list[int]
    group_labels: list[str]
    tissues: list[str]


@dataclass(frozen=True)
class TrainingEpochResult:
    """Metrics and sampling diagnostics for one training epoch."""

    mse_normalized: float
    mae_normalized: float
    mean_gradient_norm_before_clipping: float
    max_gradient_norm_before_clipping: float
    group_draw_counts: dict[str, int]
    group_unique_counts: dict[str, int]
    total_unique_records: int


@dataclass(frozen=True)
class ValidationResult:
    """Validation predictions in normalized and original target scales."""

    normalized_metrics: RegressionMetrics
    raw_metrics: RegressionMetrics
    raw_group_metrics: dict[str, RegressionMetrics]
    interaction: InteractionMetrics
    predictions_normalized: np.ndarray
    targets_normalized: np.ndarray
    predictions_raw: np.ndarray
    targets_raw: np.ndarray
    groups: list[str]
    tissues: list[str]


class DiagnosticSubset(Dataset):
    """Wrap an existing dataset and expose metadata without modifying it."""

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        group_labels: Sequence[str],
        tissues: Sequence[str],
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices]
        self.group_labels = list(group_labels)
        self.tissues = [str(tissue) for tissue in tissues]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, subset_index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        str,
        str]:
        original_index = self.indices[subset_index]
        tissue_id, dna_tokens, target = self.base_dataset[original_index]

        return (
            tissue_id,
            dna_tokens,
            target,
            original_index,
            self.group_labels[original_index],
            self.tissues[original_index])


def collate_diagnostic_batch(
    batch: list[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
            str,
            str,
        ]
    ],
    pad_id: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[str],
]:
    tissue_ids, dna_tokens, targets, indices, groups, tissues = zip(*batch)

    model_batch = list(zip(tissue_ids, dna_tokens, targets))
    tissue_batch, dna_batch, dna_mask, target_batch = (
        collate_expression_batch(model_batch, pad_id)
    )

    return (
        tissue_batch,
        dna_batch,
        dna_mask,
        target_batch,
        torch.tensor(indices, dtype=torch.long),
        list(groups),
        list(tissues),
    )

def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for comparable experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split_indices(
    group_labels: Sequence[str],
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create a deterministic split stratified by the four control groups."""

    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")
    if len(group_labels) < 2:
        raise ValueError("Dataset must contain at least two samples.")

    rng = np.random.default_rng(seed)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_labels):
        grouped[group].append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for group in sorted(grouped):
        indices = np.asarray(grouped[group], dtype=np.int64)
        rng.shuffle(indices)

        if len(indices) == 1:
            n_val = 0
        else:
            n_val = int(round(len(indices) * val_split))
            n_val = min(max(n_val, 1), len(indices) - 1)

        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())

    if not train_indices or not val_indices:
        raise ValueError(
            "The stratified split produced an empty train or validation set."
        )

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_group_balanced_sampler(
    train_groups: Sequence[str],
    seed: int,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Create inverse-frequency sampling weights for groups present in train."""

    counts = Counter(train_groups)
    if not counts:
        raise ValueError("Cannot construct a sampler for an empty training set.")

    weights = torch.tensor(
        [1.0 / counts[group] for group in train_groups], dtype=torch.double
    )
    draws = len(train_groups) if num_samples is None else int(num_samples)
    if draws < 1:
        raise ValueError("num_samples must be positive.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=draws,
        replacement=True,
        generator=generator,
    )


def build_diagnostic_dataloaders(
    dataset: Dataset,
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    pad_id: int,
    active_tissue: str,
    motif: str,
    seed: int,
    balanced_sampler: bool = False,
    samples_per_epoch: int | None = None,
) -> DiagnosticDataBundle:
    """Build deterministic train/validation loaders with train-only scaling."""

    if len(dataset) != len(dataframe):
        raise ValueError("Dataset and DataFrame must have the same length.")

    dataframe = dataframe.reset_index(drop=True)
    group_labels = build_group_labels(dataframe, active_tissue, motif)
    tissues = dataframe["tissue"].astype(str).tolist()
    train_indices, val_indices = stratified_split_indices(
        group_labels=group_labels,
        val_split=float(config["val_split"]),
        seed=seed,
    )

    train_raw_targets = np.asarray(
        [dataset.get_raw_target(index) for index in train_indices],
        dtype=np.float64,
    )
    normalizer = TargetNormalizer()
    normalizer.fit(train_raw_targets)
    dataset.attach_target_normalizer(normalizer)

    train_subset = DiagnosticSubset(
        dataset, train_indices, group_labels, tissues
    )
    val_subset = DiagnosticSubset(dataset, val_indices, group_labels, tissues)
    collate_fn = functools.partial(collate_diagnostic_batch, pad_id=pad_id)

    train_groups = [group_labels[index] for index in train_indices]
    generator = torch.Generator()
    generator.manual_seed(seed)

    if balanced_sampler:
        sampler = build_group_balanced_sampler(
            train_groups=train_groups,
            seed=seed,
            num_samples=samples_per_epoch,
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=int(config["batch_size"]),
            sampler=sampler,
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            generator=generator,
            collate_fn=collate_fn,
        )

    val_loader = DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=collate_fn,
    )

    return DiagnosticDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        train_indices=train_indices,
        val_indices=val_indices,
        group_labels=group_labels,
        tissues=tissues,
    )


def run_training_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    max_grad_norm: float | None,
    gradient_accumulation_steps: int,
) -> TrainingEpochResult:
    """Train for one epoch and report sampled groups and gradient norms."""

    model.train()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    num_samples = 0
    gradient_norms: list[float] = []
    group_draw_counts: Counter[str] = Counter()
    unique_by_group: dict[str, set[int]] = defaultdict(set)
    unique_records: set[int] = set()

    accum_steps = int(gradient_accumulation_steps)
    if accum_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    num_batches = len(train_loader)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        (
            tissue_ids,
            dna_tokens,
            dna_mask,
            targets,
            indices,
            groups,
            _,
        ) = batch

        tissue_ids = tissue_ids.to(device)
        dna_tokens = dna_tokens.to(device)
        dna_mask = dna_mask.to(device)
        targets = targets.to(device)

        predictions = model(tissue_ids, dna_tokens, dna_mask)
        loss = criterion(predictions, targets)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss encountered: {loss.item()}"
            )

        # Scale the loss so the accumulated gradient corresponds to
        # the average loss over the effective batch.
        (loss / accum_steps).backward()

        is_accumulation_step = (
            (batch_idx + 1) % accum_steps == 0
        )
        is_last_batch = (
            (batch_idx + 1) == num_batches
        )

        if is_accumulation_step or is_last_batch:
            if max_grad_norm is not None and max_grad_norm > 0:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm,
                )
                gradient_norms.append(
                    float(gradient_norm.detach().cpu())
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        errors = predictions.detach() - targets
        total_squared_error += torch.sum(errors**2).item()
        total_absolute_error += torch.sum(torch.abs(errors)).item()
        num_samples += targets.numel()

        for original_index, group in zip(indices.tolist(), groups):
            group_draw_counts[group] += 1
            unique_by_group[group].add(int(original_index))
            unique_records.add(int(original_index))

    if num_samples == 0:
        raise RuntimeError("Training loader yielded no samples.")

    mean_grad = float(np.mean(gradient_norms)) if gradient_norms else float("nan")
    max_grad = float(np.max(gradient_norms)) if gradient_norms else float("nan")

    return TrainingEpochResult(
        mse_normalized=total_squared_error / num_samples,
        mae_normalized=total_absolute_error / num_samples,
        mean_gradient_norm_before_clipping=mean_grad,
        max_gradient_norm_before_clipping=max_grad,
        group_draw_counts={
            group: group_draw_counts.get(group, 0) for group in GROUP_ORDER
        },
        group_unique_counts={
            group: len(unique_by_group.get(group, set()))
            for group in GROUP_ORDER
        },
        total_unique_records=len(unique_records),
    )


def run_validation_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    normalizer: TargetNormalizer,
) -> ValidationResult:
    """Evaluate and retain predictions for group and interaction diagnostics."""

    model.eval()
    prediction_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    groups: list[str] = []
    tissues: list[str] = []

    with torch.no_grad():
        for (tissue_ids, dna_tokens, dna_mask, targets, _, batch_groups, batch_tissues, ) in val_loader:
            tissue_ids = tissue_ids.to(device)
            dna_tokens = dna_tokens.to(device)
            dna_mask = dna_mask.to(device)
            targets = targets.to(device)
            predictions = model(tissue_ids, dna_tokens, dna_mask)

            prediction_batches.append(predictions.detach().cpu())
            target_batches.append(targets.detach().cpu())
            groups.extend(batch_groups)
            tissues.extend(batch_tissues)

    if not prediction_batches:
        raise RuntimeError("Validation loader yielded no samples.")

    predictions_normalized = torch.cat(prediction_batches).numpy()
    targets_normalized = torch.cat(target_batches).numpy()
    predictions_raw = np.asarray(
        normalizer.inverse_transform(predictions_normalized), dtype=np.float64
    )
    targets_raw = np.asarray(
        normalizer.inverse_transform(targets_normalized), dtype=np.float64
    )

    normalized_metrics = compute_regression_metrics(
        predictions_normalized, targets_normalized
    )
    raw_metrics = compute_regression_metrics(predictions_raw, targets_raw)
    raw_group_metrics = compute_group_metrics(
        predictions_raw, targets_raw, groups
    )

    return ValidationResult(
        normalized_metrics=normalized_metrics,
        raw_metrics=raw_metrics,
        raw_group_metrics=raw_group_metrics,
        interaction=compute_interaction_metrics(raw_group_metrics),
        predictions_normalized=predictions_normalized,
        targets_normalized=targets_normalized,
        predictions_raw=predictions_raw,
        targets_raw=targets_raw,
        groups=groups,
        tissues=tissues,
    )


def _format_sampled_groups(
    result: TrainingEpochResult,
    active_tissue: str,
) -> str:
    total_draws = sum(result.group_draw_counts.values())
    lines: list[str] = []
    for group in GROUP_ORDER:
        draws = result.group_draw_counts[group]
        share = 100.0 * draws / total_draws if total_draws else 0.0
        lines.append(
            f"    {display_group_name(group, active_tissue):<30} "
            f"draws={draws:5d} ({share:5.1f}%) | "
            f"unique={result.group_unique_counts[group]:5d}"
        )
    lines.append(
        f"    total draws={total_draws} | "
        f"unique training records={result.total_unique_records}"
    )
    return "\n".join(lines)


def train_model_with_diagnostics(
    model: nn.Module,
    data: DiagnosticDataBundle,
    config: dict[str, Any],
    device: str,
    active_tissue: str,
    motif: str,
    tissue_baseline_raw_mse: float,
    callback: BestCheckpointEarlyStopping,
    max_grad_norm: float = 1.0,
) -> list[dict[str, float]]:
    """Train with clipping, checkpoints, early stopping, and diagnostics."""
    accum_steps = int(config["gradient_accumulation_steps"])
    if accum_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["start_lr"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    criterion = nn.MSELoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        current_lr = get_lr_for_epoch(epoch, config)
        set_optimizer_lr(optimizer, current_lr)
        train_result = run_training_epoch(
            model=model,
            train_loader=data.train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=accum_steps,
        )
        validation = run_validation_epoch(
            model=model,
            val_loader=data.val_loader,
            device=device,
            normalizer=data.normalizer,
        )

        delta_vs_tissue = validation.raw_metrics.mse - tissue_baseline_raw_mse
        callback_result = callback.step(
            monitored_value=validation.normalized_metrics.mse,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            extra_state={
                "target_normalizer": data.normalizer.to_dict(),
                "active_tissue": active_tissue,
                "motif": motif,
                "validation_mse_raw": validation.raw_metrics.mse,
                "interaction_prediction": (
                    validation.interaction.interaction_prediction
                ),
                "interaction_target": validation.interaction.interaction_target,
            },
        )

        saved = " | saved" if callback_result.improved else ""
        print(
            f"Epoch {epoch:03d}/{int(config['epochs']):03d} | "
            f"train MSE(norm)={train_result.mse_normalized:.6f} | "
            f"val MSE(norm)={validation.normalized_metrics.mse:.6f} | "
            f"val MSE(raw)={validation.raw_metrics.mse:.6f} | "
            f"LR={current_lr:.2e} | "
            f"delta vs tissue-only={delta_vs_tissue:+.6f}{saved}"
        )
        print(
            "  gradients before clipping: "
            f"mean norm={train_result.mean_gradient_norm_before_clipping:.4f}, "
            f"max norm={train_result.max_gradient_norm_before_clipping:.4f}, "
            f"clip={max_grad_norm:.4f}"
        )
        print("  sampled training groups:")
        print(_format_sampled_groups(train_result, active_tissue))
        print("  validation groups in original target scale:")
        print(
            format_group_metrics(
                validation.raw_group_metrics,
                active_tissue=active_tissue,
                indent="    ",
            )
        )
        interaction = validation.interaction
        print(
            "  tissue x motif interaction (difference-in-differences): "
            f"prediction={interaction.interaction_prediction:.4f}, "
            f"target={interaction.interaction_target:.4f}"
        )
        print(
            "  motif effect: "
            f"{active_tissue} prediction="
            f"{interaction.active_tissue_motif_effect_prediction:.4f}, "
            "other tissues prediction="
            f"{interaction.other_tissues_motif_effect_prediction:.4f}"
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_mse_normalized": train_result.mse_normalized,
                "validation_mse_normalized": validation.normalized_metrics.mse,
                "validation_mse_raw": validation.raw_metrics.mse,
                "delta_vs_tissue_baseline_raw": delta_vs_tissue,
                "interaction_prediction": interaction.interaction_prediction,
                "interaction_target": interaction.interaction_target,
            }
        )

        if callback_result.should_stop:
            print(
                "Early stopping: no sufficient validation improvement for "
                f"{callback.patience} epochs."
            )
            break

    print(
        f"Best validation MSE (normalized): {callback.best_value:.6f} "
        f"at epoch {callback.best_epoch}."
    )
    print(f"Best checkpoint: {callback.checkpoint_path}")
    return history
