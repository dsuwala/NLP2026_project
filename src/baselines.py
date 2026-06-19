#!/usr/bin/env python3
"""Simple control baselines for genomic expression regression."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class ConstantMeanBaseline:
    """Predict the global mean target observed in the training split."""

    mean_: float | None = None

    def fit(self, targets: Sequence[float] | np.ndarray) -> "ConstantMeanBaseline":
        values = np.asarray(targets, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError("Cannot fit baseline on an empty target array.")
        self.mean_ = float(np.mean(values))
        return self

    def predict(self, n_samples: int) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("ConstantMeanBaseline must be fitted first.")
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative.")
        return np.full(n_samples, self.mean_, dtype=np.float64)


@dataclass
class TissueMeanBaseline:
    """Predict a train-set mean for each tissue while ignoring DNA sequence."""

    tissue_means_: dict[str, float] = field(default_factory=dict)
    global_mean_: float | None = None

    def fit(
        self,
        tissues: Sequence[str],
        targets: Sequence[float] | np.ndarray,
    ) -> "TissueMeanBaseline":
        tissue_array = np.asarray([str(t) for t in tissues], dtype=object)
        target_array = np.asarray(targets, dtype=np.float64).reshape(-1)

        if tissue_array.size != target_array.size:
            raise ValueError("Tissues and targets must have equal lengths.")
        if target_array.size == 0:
            raise ValueError("Cannot fit baseline on an empty training split.")

        self.global_mean_ = float(np.mean(target_array))
        self.tissue_means_.clear()
        for tissue in np.unique(tissue_array):
            mask = tissue_array == tissue
            self.tissue_means_[str(tissue)] = float(np.mean(target_array[mask]))
        return self

    def predict(self, tissues: Sequence[str]) -> np.ndarray:
        if self.global_mean_ is None:
            raise RuntimeError("TissueMeanBaseline must be fitted first.")
        return np.asarray(
            [
                self.tissue_means_.get(str(tissue), self.global_mean_)
                for tissue in tissues
            ],
            dtype=np.float64,
        )
