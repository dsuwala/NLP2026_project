#!/usr/bin/env python3
"""Checkpointing and early-stopping utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CallbackResult:
    """Outcome of one checkpoint/early-stopping update."""

    improved: bool
    should_stop: bool
    epochs_without_improvement: int


class BestCheckpointEarlyStopping:
    """Save the best model and stop after a validation plateau.

    The monitored quantity is minimized.  A new value counts as an improvement
    only when it is lower than ``best_value - min_delta``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        patience: int = 10,
        min_delta: float = 0.0,
        start_epoch: int = 1,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative.")
        if start_epoch < 1:
            raise ValueError("start_epoch must be at least 1.")

        self.checkpoint_path = Path(checkpoint_path)
        self.patience = patience
        self.min_delta = min_delta
        self.start_epoch = start_epoch

        # Global best, used for checkpointing.
        self.best_value = float("inf")
        self.best_epoch: int | None = None

        # Separate state used only for delayed early stopping.
        self.early_stopping_best_value = float("inf")
        self.epochs_without_improvement = 0

    def _save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        monitored_value: float,
        config: Mapping[str, Any],
        extra_state: Mapping[str, Any] | None,
    ) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "epoch": epoch,
            "monitored_value": monitored_value,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dict(config),
        }
        if extra_state:
            payload.update(dict(extra_state))
        torch.save(payload, self.checkpoint_path)

    def step(
        self,
        monitored_value: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        config: Mapping[str, Any],
        extra_state: Mapping[str, Any] | None = None,
    ) -> CallbackResult:
        """Update checkpoint and early-stopping state after one epoch."""

        if not torch.isfinite(torch.tensor(monitored_value)):
            raise ValueError(
                f"Monitored validation value is not finite: {monitored_value}"
            )

        improved = monitored_value < self.best_value - self.min_delta
        if improved:
            self.best_value = float(monitored_value)
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            self._save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                monitored_value=monitored_value,
                config=config,
                extra_state=extra_state,
            )
        if epoch < self.start_epoch:
            self.epochs_without_improvement = 0
            should_stop = False
        else:
            early_stopping_improved = (monitored_value < self.early_stopping_best_value - self.min_delta)

            if early_stopping_improved:
                self.early_stopping_best_value = float(monitored_value)
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            should_stop = (self.epochs_without_improvement >= self.patience)

        return CallbackResult(improved=improved, should_stop=should_stop, epochs_without_improvement=self.epochs_without_improvement)
