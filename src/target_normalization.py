#!/usr/bin/env python3
"""
Reversible target normalization for VST expression regression.

Applies train-only min-max scaling to [0, 1] followed by z-score standardization.
Statistics are persisted to JSON for inverse transform at inference time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TargetNormalizer:
    """Two-step target scaler fit on training data only.

    Forward: min-max to [0, 1] using train min/max, then z-score using train
    mean/std of the min-max values. Inverse restores original VST units.

    Attributes:
        min_val: Minimum raw target observed in the training split.
        max_val: Maximum raw target observed in the training split.
        mean: Mean of min-max scaled training targets.
        std: Standard deviation of min-max scaled training targets.
    """

    min_val: float = 0.0
    max_val: float = 0.0
    mean: float = 0.0
    std: float = 1.0

    def fit(self, raw_targets: np.ndarray) -> None:
        """Compute normalization statistics from raw training targets only.

        Args:
            raw_targets: 1-D array of unnormalized VST expression values.

        Raises:
            ValueError: If all training targets are identical.
        """
        raw = np.asarray(raw_targets, dtype=np.float64).ravel()
        self.min_val = float(raw.min())
        self.max_val = float(raw.max())

        if self.max_val == self.min_val:
            raise ValueError("Cannot normalize constant targets.")

        minmax = (raw - self.min_val) / (self.max_val - self.min_val)
        self.mean = float(minmax.mean())
        self.std = float(minmax.std())

        if self.std == 0.0:
            self.std = 1.0

    def transform(self, raw: np.ndarray | float) -> np.ndarray:
        """Apply min-max then z-score normalization.

        Args:
            raw: Scalar or array of raw VST targets.

        Returns:
            Normalized values with the same shape as the input.
        """
        arr = np.asarray(raw, dtype=np.float64)
        minmax = (arr - self.min_val) / (self.max_val - self.min_val)
        return (minmax - self.mean) / self.std

    def inverse_transform(self, normalized: np.ndarray | float) -> np.ndarray:
        """Reverse z-score and min-max to recover original VST units.

        Args:
            normalized: Scalar or array of model outputs in normalized space.

        Returns:
            Values in original VST units with the same shape as the input.
        """
        arr = np.asarray(normalized, dtype=np.float64)
        minmax = arr * self.std + self.mean
        return minmax * (self.max_val - self.min_val) + self.min_val

    def transform_tensor(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply normalization to a PyTorch tensor.

        Args:
            raw: Tensor of raw VST targets.

        Returns:
            Tensor in normalized space.
        """
        minmax = (raw - self.min_val) / (self.max_val - self.min_val)
        return (minmax - self.mean) / self.std

    def inverse_transform_tensor(self, normalized: torch.Tensor) -> torch.Tensor:
        """Reverse normalization on a PyTorch tensor.

        Args:
            normalized: Tensor of model predictions in normalized space.

        Returns:
            Tensor in original VST units.
        """
        minmax = normalized * self.std + self.mean
        return minmax * (self.max_val - self.min_val) + self.min_val

    def to_dict(self) -> dict[str, float]:
        """Serialize fitted statistics to a plain dictionary.

        Returns:
            Dict with keys ``min_val``, ``max_val``, ``mean``, ``std``.
        """
        return {
            "min_val": self.min_val,
            "max_val": self.max_val,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> TargetNormalizer:
        """Restore a normalizer from a serialized dictionary.

        Args:
            data: Dict produced by ``to_dict``.

        Returns:
            Fitted ``TargetNormalizer`` instance.
        """
        return cls(
            min_val=float(data["min_val"]),
            max_val=float(data["max_val"]),
            mean=float(data["mean"]),
            std=float(data["std"]),
        )

    def save(self, path: str) -> None:
        """Write fitted statistics to a JSON file.

        Args:
            path: Output file path.
        """
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)

    @classmethod
    def load(cls, path: str) -> TargetNormalizer:
        """Load fitted statistics from a JSON file.

        Args:
            path: Path to a file written by ``save``.

        Returns:
            Restored ``TargetNormalizer`` instance.
        """
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        return cls.from_dict(data)
