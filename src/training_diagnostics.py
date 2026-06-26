#!/usr/bin/env python3
"""Training utilities for the no-CNN Transformer with leakage-safe diagnostics.

The module provides:

* gene-level (or other group-level) train/validation splitting;
* stratification by visible motif status and active-tissue availability;
* four diagnostic groups based on the sequence prefix actually seen by the model;
* train-only target normalization;
* LR warmup, gradient accumulation, clipping, checkpointing and early stopping.

It expects the no-CNN interface:

* dataset item: ``(tokens, target)``;
* model call: ``model(tokens, key_padding_mask)``.
"""
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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from callbacks import BestCheckpointEarlyStopping
from diagnostics import (
    ACTIVE_WITH_MOTIF,
    ACTIVE_WITHOUT_MOTIF,
    GROUP_ORDER,
    OTHER_WITH_MOTIF,
    InteractionMetrics,
    RegressionMetrics,
    classify_sample,
    compute_group_metrics,
    compute_interaction_metrics,
    compute_regression_metrics,
    count_groups,
    display_group_name,
    format_group_metrics,
)
from target_normalization import TargetNormalizer
from training import get_lr_for_epoch, set_optimizer_lr


@dataclass(frozen=True)
class DiagnosticDataBundle:
    """DataLoaders, split metadata and the train-fitted target normalizer."""

    train_loader: DataLoader
    val_loader: DataLoader
    normalizer: TargetNormalizer
    train_indices: list[int]
    val_indices: list[int]
    group_labels: list[str]
    tissues: list[str]
    split_group_column: str
    train_split_groups: list[str]
    val_split_groups: list[str]
    max_visible_dna_len: int
    truncated_record_count: int
    full_motif_count: int
    visible_motif_count: int
    motif_lost_to_truncation_count: int


@dataclass(frozen=True)
class TrainingEpochResult:
    """Sample-weighted metrics and sampling diagnostics for one epoch."""

    mse_normalized: float
    mae_normalized: float
    mean_gradient_norm_before_clipping: float
    max_gradient_norm_before_clipping: float
    group_draw_counts: dict[str, int]
    group_unique_counts: dict[str, int]
    total_unique_records: int


@dataclass(frozen=True)
class ValidationResult:
    """Validation metrics in normalized and original target scales."""

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
    """Wrap a dataset while retaining original indices and diagnostic labels."""

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
        self,
        subset_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, str, str]:
        original_index = self.indices[subset_index]
        tokens, target = self.base_dataset[original_index]
        return (
            tokens,
            target,
            original_index,
            self.group_labels[original_index],
            self.tissues[original_index],
        )


def collate_diagnostic_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, int, str, str]],
    pad_id: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    list[str],
]:
    """Pad sequences while retaining indices and diagnostic metadata."""

    if not batch:
        raise ValueError("Cannot collate an empty diagnostic batch.")

    tokens, targets, indices, groups, tissues = zip(*batch)
    padded = pad_sequence(tokens, batch_first=True, padding_value=pad_id)
    key_padding_mask = padded.eq(pad_id)
    return (
        padded,
        key_padding_mask,
        torch.stack(targets, dim=0),
        torch.tensor(indices, dtype=torch.long),
        list(groups),
        list(tissues),
    )


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for comparable experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_visible_group_labels(
    dataset: Dataset,
    dataframe: pd.DataFrame,
    active_tissue: str,
    motif: str,
) -> tuple[list[str], int, int, int, int, int]:
    """Classify rows using the DNA prefix visible after dataset truncation."""

    required_methods = (
        "get_full_dna_sequence",
        "get_visible_dna_sequence",
        "max_visible_dna_len",
    )
    missing = [name for name in required_methods if not hasattr(dataset, name)]
    if missing:
        raise TypeError(
            "Dataset does not expose the truncation-aware diagnostic API: "
            f"missing {missing}."
        )

    motif_upper = motif.upper()
    labels: list[str] = []
    truncated_count = 0
    full_motif_count = 0
    visible_motif_count = 0
    lost_motif_count = 0

    for index, tissue in enumerate(dataframe["tissue"].astype(str)):
        full_sequence = dataset.get_full_dna_sequence(index)
        visible_sequence = dataset.get_visible_dna_sequence(index)

        full_has_motif = motif_upper in full_sequence.upper()
        visible_has_motif = motif_upper in visible_sequence.upper()

        if len(full_sequence) > int(dataset.max_visible_dna_len):
            truncated_count += 1
        if full_has_motif:
            full_motif_count += 1
        if visible_has_motif:
            visible_motif_count += 1
        if full_has_motif and not visible_has_motif:
            lost_motif_count += 1

        labels.append(
            classify_sample(
                tissue=tissue,
                sequence=visible_sequence,
                active_tissue=active_tissue,
                motif=motif,
            )
        )

    return (
        labels,
        int(dataset.max_visible_dna_len),
        truncated_count,
        full_motif_count,
        visible_motif_count,
        lost_motif_count,
    )


def grouped_stratified_split_indices(
    dataframe: pd.DataFrame,
    group_labels: Sequence[str],
    split_group_column: str,
    val_split: float,
    seed: int,
) -> tuple[list[int], list[int], list[str], list[str]]:
    """Split complete groups while approximately preserving motif strata.

    All rows sharing ``split_group_column`` are assigned to exactly one split.
    Split groups are stratified by two binary properties derived from their rows:
    whether the active tissue is represented and whether a visible motif is
    represented. For the integrated dataset this becomes a gene-level split
    stratified by visible motif presence.
    """

    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1.")
    if split_group_column not in dataframe.columns:
        raise ValueError(
            f"Split group column {split_group_column!r} is absent from the data."
        )
    if dataframe[split_group_column].isna().any():
        raise ValueError(
            f"Split group column {split_group_column!r} contains missing values."
        )
    if len(group_labels) != len(dataframe):
        raise ValueError("group_labels and dataframe must have the same length.")

    split_values = dataframe[split_group_column].astype(str).tolist()
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, split_value in enumerate(split_values):
        group_to_indices[split_value].append(index)

    if len(group_to_indices) < 2:
        raise ValueError(
            "Grouped splitting requires at least two distinct split groups."
        )

    motif_groups = {ACTIVE_WITH_MOTIF, OTHER_WITH_MOTIF}
    active_groups = {ACTIVE_WITH_MOTIF, ACTIVE_WITHOUT_MOTIF}
    strata: dict[tuple[bool, bool], list[str]] = defaultdict(list)

    for split_value, indices in group_to_indices.items():
        labels = {group_labels[index] for index in indices}
        has_active_tissue = bool(labels & active_groups)
        has_visible_motif = bool(labels & motif_groups)
        strata[(has_active_tissue, has_visible_motif)].append(split_value)

    rng = np.random.default_rng(seed)
    train_groups: list[str] = []
    val_groups: list[str] = []

    for stratum in sorted(strata):
        values = np.asarray(sorted(strata[stratum]), dtype=object)
        rng.shuffle(values)

        if len(values) == 1:
            n_val = 0
        else:
            n_val = int(round(len(values) * val_split))
            n_val = min(max(n_val, 1), len(values) - 1)

        val_groups.extend(str(value) for value in values[:n_val])
        train_groups.extend(str(value) for value in values[n_val:])

    # Rare singleton-only strata can otherwise leave validation empty.
    if not val_groups:
        all_groups = np.asarray(sorted(group_to_indices), dtype=object)
        rng.shuffle(all_groups)
        n_val = int(round(len(all_groups) * val_split))
        n_val = min(max(n_val, 1), len(all_groups) - 1)
        val_groups = [str(value) for value in all_groups[:n_val]]
        train_groups = [str(value) for value in all_groups[n_val:]]

    train_group_set = set(train_groups)
    val_group_set = set(val_groups)
    overlap = train_group_set & val_group_set
    if overlap:
        raise RuntimeError(
            "Grouped split leakage detected for split groups: "
            f"{sorted(overlap)[:5]}"
        )

    train_indices = [
        index
        for split_value in train_groups
        for index in group_to_indices[split_value]
    ]
    val_indices = [
        index
        for split_value in val_groups
        for index in group_to_indices[split_value]
    ]

    if not train_indices or not val_indices:
        raise ValueError(
            "Grouped split produced an empty training or validation set."
        )

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(train_groups)
    rng.shuffle(val_groups)
    return train_indices, val_indices, train_groups, val_groups


def build_group_balanced_sampler(
    train_groups: Sequence[str],
    seed: int,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Create inverse-frequency sampling weights for diagnostic row groups."""

    counts = Counter(train_groups)
    if not counts:
        raise ValueError("Cannot construct a sampler for an empty training set.")

    weights = torch.tensor(
        [1.0 / counts[group] for group in train_groups],
        dtype=torch.double,
    )
    draws = len(train_groups) if num_samples is None else int(num_samples)
    if draws < 1:
        raise ValueError("samples_per_epoch must be positive.")

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
    split_group_column: str = "gene_id",
    balanced_sampler: bool = False,
    samples_per_epoch: int | None = None,
) -> DiagnosticDataBundle:
    """Build leakage-safe loaders and fit target normalization on train only."""

    if len(dataset) != len(dataframe):
        raise ValueError("Dataset and DataFrame must have the same length.")

    dataframe = dataframe.reset_index(drop=True)
    tissues = dataframe["tissue"].astype(str).tolist()
    (
        group_labels,
        max_visible_dna_len,
        truncated_count,
        full_motif_count,
        visible_motif_count,
        lost_motif_count,
    ) = _build_visible_group_labels(
        dataset=dataset,
        dataframe=dataframe,
        active_tissue=active_tissue,
        motif=motif,
    )

    (
        train_indices,
        val_indices,
        train_split_groups,
        val_split_groups,
    ) = grouped_stratified_split_indices(
        dataframe=dataframe,
        group_labels=group_labels,
        split_group_column=split_group_column,
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
        dataset,
        train_indices,
        group_labels,
        tissues,
    )
    val_subset = DiagnosticSubset(
        dataset,
        val_indices,
        group_labels,
        tissues,
    )
    collate_fn = functools.partial(collate_diagnostic_batch, pad_id=pad_id)

    train_row_groups = [group_labels[index] for index in train_indices]
    generator = torch.Generator()
    generator.manual_seed(seed)

    if balanced_sampler:
        sampler = build_group_balanced_sampler(
            train_groups=train_row_groups,
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
        split_group_column=split_group_column,
        train_split_groups=train_split_groups,
        val_split_groups=val_split_groups,
        max_visible_dna_len=max_visible_dna_len,
        truncated_record_count=truncated_count,
        full_motif_count=full_motif_count,
        visible_motif_count=visible_motif_count,
        motif_lost_to_truncation_count=lost_motif_count,
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
    """Train for one epoch with accumulation and optional gradient clipping."""

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")

    model.train()
    total_squared_error = 0.0
    total_absolute_error = 0.0
    num_samples = 0
    gradient_norms: list[float] = []
    group_draw_counts: Counter[str] = Counter()
    unique_by_group: dict[str, set[int]] = defaultdict(set)
    unique_records: set[int] = set()

    num_batches = len(train_loader)
    if num_batches == 0:
        raise RuntimeError("Training loader yielded no batches.")

    optimizer.zero_grad(set_to_none=True)

    for batch_index, (
        sequences,
        key_padding_mask,
        targets,
        indices,
        groups,
        _,
    ) in enumerate(train_loader):
        sequences = sequences.to(device)
        key_padding_mask = key_padding_mask.to(device)
        targets = targets.to(device)

        predictions = model(sequences, key_padding_mask)
        loss = criterion(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss encountered: {loss}")

        window_start = (
            batch_index // gradient_accumulation_steps
        ) * gradient_accumulation_steps
        window_size = min(
            gradient_accumulation_steps,
            num_batches - window_start,
        )
        (loss / window_size).backward()

        is_accumulation_boundary = (
            (batch_index + 1) % gradient_accumulation_steps == 0
        )
        is_last_batch = (batch_index + 1) == num_batches
        if is_accumulation_boundary or is_last_batch:
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

    mean_grad = (
        float(np.mean(gradient_norms))
        if gradient_norms
        else float("nan")
    )
    max_grad = (
        float(np.max(gradient_norms))
        if gradient_norms
        else float("nan")
    )

    return TrainingEpochResult(
        mse_normalized=total_squared_error / num_samples,
        mae_normalized=total_absolute_error / num_samples,
        mean_gradient_norm_before_clipping=mean_grad,
        max_gradient_norm_before_clipping=max_grad,
        group_draw_counts={
            group: group_draw_counts.get(group, 0)
            for group in GROUP_ORDER
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
    """Evaluate validation data and retain group-level predictions."""

    model.eval()
    prediction_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    groups: list[str] = []
    tissues: list[str] = []

    with torch.no_grad():
        for (
            sequences,
            key_padding_mask,
            targets,
            _,
            batch_groups,
            batch_tissues,
        ) in val_loader:
            sequences = sequences.to(device)
            key_padding_mask = key_padding_mask.to(device)
            targets = targets.to(device)

            predictions = model(sequences, key_padding_mask)
            prediction_batches.append(predictions.detach().cpu())
            target_batches.append(targets.detach().cpu())
            groups.extend(batch_groups)
            tissues.extend(batch_tissues)

    if not prediction_batches:
        raise RuntimeError("Validation loader yielded no samples.")

    predictions_normalized = torch.cat(prediction_batches).numpy()
    targets_normalized = torch.cat(target_batches).numpy()
    predictions_raw = np.asarray(
        normalizer.inverse_transform(predictions_normalized),
        dtype=np.float64,
    )
    targets_raw = np.asarray(
        normalizer.inverse_transform(targets_normalized),
        dtype=np.float64,
    )

    normalized_metrics = compute_regression_metrics(
        predictions_normalized,
        targets_normalized,
    )
    raw_metrics = compute_regression_metrics(predictions_raw, targets_raw)
    raw_group_metrics = compute_group_metrics(
        predictions_raw,
        targets_raw,
        groups,
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
            f"draws={draws:7d} ({share:5.1f}%) | "
            f"unique={result.group_unique_counts[group]:7d}"
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
    max_grad_norm: float | None = 1.0,
) -> list[dict[str, float]]:
    """Train with the project LR schedule and report diagnostic metrics."""

    accumulation_steps = int(config["gradient_accumulation_steps"])
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["start_lr"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    criterion = nn.MSELoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, int(config["epochs"]) + 1):
        current_lr = float(get_lr_for_epoch(epoch, config))
        set_optimizer_lr(optimizer, current_lr)

        train_result = run_training_epoch(
            model=model,
            train_loader=data.train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=accumulation_steps,
        )
        validation = run_validation_epoch(
            model=model,
            val_loader=data.val_loader,
            device=device,
            normalizer=data.normalizer,
        )

        delta_vs_tissue = (
            validation.raw_metrics.mse - tissue_baseline_raw_mse
        )
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
                "split_group_column": data.split_group_column,
                "validation_mse_raw": validation.raw_metrics.mse,
                "interaction_prediction": (
                    validation.interaction.interaction_prediction
                ),
                "interaction_target": (
                    validation.interaction.interaction_target
                ),
            },
        )

        saved = " | saved" if callback_result.improved else ""
        print(
            f"Epoch {epoch:03d}/{int(config['epochs']):03d} | "
            f"LR={current_lr:.2e} | "
            f"train MSE(norm)={train_result.mse_normalized:.6f} | "
            f"val MSE(norm)={validation.normalized_metrics.mse:.6f} | "
            f"val MSE(raw)={validation.raw_metrics.mse:.6f} | "
            f"delta vs tissue-only={delta_vs_tissue:+.6f}{saved}"
        )
        print(
            "  gradients before clipping: "
            f"mean norm={train_result.mean_gradient_norm_before_clipping:.4f}, "
            f"max norm={train_result.max_gradient_norm_before_clipping:.4f}, "
            f"clip={max_grad_norm}"
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
                "learning_rate": current_lr,
                "train_mse_normalized": train_result.mse_normalized,
                "validation_mse_normalized": (
                    validation.normalized_metrics.mse
                ),
                "validation_mse_raw": validation.raw_metrics.mse,
                "delta_vs_tissue_baseline_raw": delta_vs_tissue,
                "interaction_prediction": (
                    interaction.interaction_prediction
                ),
                "interaction_target": interaction.interaction_target,
            }
        )

        if callback_result.should_stop:
            print(
                "Early stopping: no sufficient validation improvement for "
                f"{callback.patience} epochs."
            )
            break

    if callback.best_epoch is None:
        raise RuntimeError("Training finished without saving a valid checkpoint.")

    print(
        f"Best validation MSE (normalized): {callback.best_value:.6f} "
        f"at epoch {callback.best_epoch}."
    )
    print(f"Best checkpoint: {callback.checkpoint_path}")
    return history
