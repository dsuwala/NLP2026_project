#!/usr/bin/env python3
"""Diagnostics for tissue-conditioned motif regression experiments.

The module defines four control groups and computes both ordinary regression
metrics and a difference-in-differences interaction score.  It is independent
of the model implementation and can therefore be reused by different training
pipelines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ACTIVE_WITH_MOTIF = "active_tissue_with_motif"
ACTIVE_WITHOUT_MOTIF = "active_tissue_without_motif"
OTHER_WITH_MOTIF = "other_tissue_with_motif"
OTHER_WITHOUT_MOTIF = "other_tissue_without_motif"

GROUP_ORDER: tuple[str, ...] = (
    ACTIVE_WITH_MOTIF,
    ACTIVE_WITHOUT_MOTIF,
    OTHER_WITH_MOTIF,
    OTHER_WITHOUT_MOTIF,
)


@dataclass(frozen=True)
class RegressionMetrics:
    """Aggregate regression metrics for one set of samples."""

    n: int
    mse: float
    rmse: float
    mae: float
    mean_prediction: float
    mean_target: float


@dataclass(frozen=True)
class InteractionMetrics:
    """Predicted and target tissue-by-motif interaction effects.

    The interaction is a difference-in-differences:

        motif effect in active tissue - motif effect in other tissues

    A model using the intended AND rule should have a positive predicted
    interaction close to the corresponding target interaction.
    """

    active_tissue_motif_effect_prediction: float
    other_tissues_motif_effect_prediction: float
    interaction_prediction: float
    active_tissue_motif_effect_target: float
    other_tissues_motif_effect_target: float
    interaction_target: float


def combine_sequence(row: pd.Series) -> str:
    """Return promoter and 5' UTR as one uppercase sequence."""

    return (
        str(row["promoter_sequence"]) + str(row["utr_5_sequence"])
    ).upper()


def classify_sample(
    tissue: str,
    sequence: str,
    active_tissue: str,
    motif: str,
) -> str:
    """Assign one sample to one of the four diagnostic groups."""

    tissue_is_active = tissue.upper() == active_tissue.upper()
    contains_motif = motif.upper() in sequence.upper()

    if tissue_is_active and contains_motif:
        return ACTIVE_WITH_MOTIF
    if tissue_is_active and not contains_motif:
        return ACTIVE_WITHOUT_MOTIF
    if not tissue_is_active and contains_motif:
        return OTHER_WITH_MOTIF
    return OTHER_WITHOUT_MOTIF


def build_group_labels(
    dataframe: pd.DataFrame,
    active_tissue: str,
    motif: str,
) -> list[str]:
    """Build four-group labels in DataFrame row order."""

    required = {"tissue", "promoter_sequence", "utr_5_sequence"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            "Cannot build diagnostic groups; missing columns: "
            f"{sorted(missing)}"
        )

    labels: list[str] = []
    for _, row in dataframe.iterrows():
        labels.append(
            classify_sample(
                tissue=str(row["tissue"]),
                sequence=combine_sequence(row),
                active_tissue=active_tissue,
                motif=motif,
            )
        )
    return labels


def display_group_name(group: str, active_tissue: str) -> str:
    """Return a readable label for console reports."""

    names = {
        ACTIVE_WITH_MOTIF: f"{active_tissue} + motif",
        ACTIVE_WITHOUT_MOTIF: f"{active_tissue} without motif",
        OTHER_WITH_MOTIF: "other tissue + motif",
        OTHER_WITHOUT_MOTIF: "other tissue without motif",
    }
    return names.get(group, group)


def count_groups(groups: Iterable[str]) -> dict[str, int]:
    """Count samples in all standard groups, including zero-count groups."""

    counts = {group: 0 for group in GROUP_ORDER}
    for group in groups:
        counts[group] = counts.get(group, 0) + 1
    return counts


def _to_1d_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array


def compute_regression_metrics(
    predictions: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
) -> RegressionMetrics:
    """Compute sample-weighted MSE, RMSE, MAE, and means."""

    prediction_array = _to_1d_float_array(predictions)
    target_array = _to_1d_float_array(targets)

    if prediction_array.shape != target_array.shape:
        raise ValueError(
            "Predictions and targets must have the same shape: "
            f"{prediction_array.shape} vs {target_array.shape}."
        )
    if prediction_array.size == 0:
        nan = float("nan")
        return RegressionMetrics(0, nan, nan, nan, nan, nan)

    errors = prediction_array - target_array
    mse = float(np.mean(errors**2))
    return RegressionMetrics(
        n=int(prediction_array.size),
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=float(np.mean(np.abs(errors))),
        mean_prediction=float(np.mean(prediction_array)),
        mean_target=float(np.mean(target_array)),
    )


def compute_group_metrics(
    predictions: Sequence[float] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    groups: Sequence[str],
) -> dict[str, RegressionMetrics]:
    """Compute regression metrics separately for each diagnostic group."""

    prediction_array = _to_1d_float_array(predictions)
    target_array = _to_1d_float_array(targets)
    group_array = np.asarray(list(groups), dtype=object).reshape(-1)

    if not (
        prediction_array.size == target_array.size == group_array.size
    ):
        raise ValueError(
            "Predictions, targets, and groups must contain the same number "
            "of samples."
        )

    result: dict[str, RegressionMetrics] = {}
    ordered_groups = list(GROUP_ORDER)
    ordered_groups.extend(
        group for group in dict.fromkeys(group_array.tolist())
        if group not in GROUP_ORDER
    )

    for group in ordered_groups:
        mask = group_array == group
        result[group] = compute_regression_metrics(
            prediction_array[mask], target_array[mask]
        )
    return result


def _difference(left: float, right: float) -> float:
    if np.isnan(left) or np.isnan(right):
        return float("nan")
    return left - right


def compute_interaction_metrics(
    group_metrics: Mapping[str, RegressionMetrics],
) -> InteractionMetrics:
    """Compute the tissue-by-motif difference-in-differences score."""

    required = GROUP_ORDER
    missing = [group for group in required if group not in group_metrics]
    if missing:
        raise ValueError(f"Missing group metrics: {missing}")

    active_with = group_metrics[ACTIVE_WITH_MOTIF]
    active_without = group_metrics[ACTIVE_WITHOUT_MOTIF]
    other_with = group_metrics[OTHER_WITH_MOTIF]
    other_without = group_metrics[OTHER_WITHOUT_MOTIF]

    active_prediction_effect = _difference(
        active_with.mean_prediction, active_without.mean_prediction
    )
    other_prediction_effect = _difference(
        other_with.mean_prediction, other_without.mean_prediction
    )
    active_target_effect = _difference(
        active_with.mean_target, active_without.mean_target
    )
    other_target_effect = _difference(
        other_with.mean_target, other_without.mean_target
    )

    return InteractionMetrics(
        active_tissue_motif_effect_prediction=active_prediction_effect,
        other_tissues_motif_effect_prediction=other_prediction_effect,
        interaction_prediction=_difference(
            active_prediction_effect, other_prediction_effect
        ),
        active_tissue_motif_effect_target=active_target_effect,
        other_tissues_motif_effect_target=other_target_effect,
        interaction_target=_difference(active_target_effect, other_target_effect),
    )


def format_group_metrics(
    group_metrics: Mapping[str, RegressionMetrics],
    active_tissue: str,
    indent: str = "  ",
) -> str:
    """Format per-group metrics for terminal logs."""

    lines: list[str] = []
    for group in GROUP_ORDER:
        metrics = group_metrics[group]
        name = display_group_name(group, active_tissue)
        lines.append(
            f"{indent}{name:<30} n={metrics.n:5d} | "
            f"prediction={metrics.mean_prediction:8.4f} | "
            f"target={metrics.mean_target:8.4f} | "
            f"mse={metrics.mse:10.4f}"
        )
    return "\n".join(lines)
