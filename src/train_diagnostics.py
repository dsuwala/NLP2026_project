#!/usr/bin/env python3
"""CLI entry point for diagnostic Transformer training.

This driver reuses the project's existing data loader, vocabulary, dataset, and
model while adding stratified diagnostics, control baselines, gradient clipping,
best-checkpoint saving, early stopping, and optional group-balanced sampling.
"""
from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from baselines import ConstantMeanBaseline, TissueMeanBaseline
from callbacks import BestCheckpointEarlyStopping
from data_utils import extract_unique_tissues, load_expression_dataframe
from dataset import ExpressionDataset
from diagnostics import (
    GROUP_ORDER,
    compute_group_metrics,
    compute_regression_metrics,
    count_groups,
    display_group_name,
    format_group_metrics,
)
from model import ExpressionTransformer
from train import CONFIG as PROJECT_CONFIG
from training import resolve_device
from training_diagnostics import (
    build_diagnostic_dataloaders,
    set_global_seed,
    train_model_with_diagnostics,
)
from vocabulary import build_vocabulary


class TeeStream:
    """Duplicate stdout writes to both terminal and a log file."""

    def __init__(self, log_file_path: str) -> None:
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        self._terminal = sys.stdout
        self._log = open(log_file_path, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self._terminal.write(message)
        self._log.write(message)
        self._terminal.flush()
        self._log.flush()

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def close(self) -> None:
        self._log.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the expression Transformer with control diagnostics."
    )
    parser.add_argument("--data", required=True, help="Input CSV/TSV dataset.")
    parser.add_argument(
        "--active-tissue",
        default="HEAD",
        help="Tissue participating in the synthetic AND rule (default: HEAD).",
    )
    parser.add_argument(
        "--motif",
        default="GATTACAA",
        help="Sequence motif used for four-group diagnostics.",
    )
    parser.add_argument(
        "--checkpoint-file",
        default="checkpoints/best_diagnostic_model.pt",
        help="Best-model checkpoint path.",
    )
    parser.add_argument(
        "--normalizer-file",
        default="checkpoints/diagnostic_target_normalizer.json",
        help="Path for the train-fitted target normalizer.",
    )
    parser.add_argument(
        "--log-file",
        default="diagnostic_training_logs.txt",
        help="Path for console logs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help="Use inverse-frequency group sampling on the training split only.",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Number of sampler draws per epoch; default is train-set size.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--device",
        default=None,
        help="Override CONFIG device, e.g. cuda, cuda:0, or cpu.",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(PROJECT_CONFIG)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr
    if args.device is not None:
        config["device"] = args.device
    config["weight_decay"] = args.weight_decay
    return config


def _print_group_distribution(
    all_groups: list[str],
    train_indices: list[int],
    val_indices: list[int],
    active_tissue: str,
) -> None:
    total_counts = count_groups(all_groups)
    train_counts = count_groups(all_groups[index] for index in train_indices)
    val_counts = count_groups(all_groups[index] for index in val_indices)
    print("Diagnostic groups:")
    for group in GROUP_ORDER:
        print(
            f"  {display_group_name(group, active_tissue):<30} "
            f"{total_counts[group]:5d} total | "
            f"{train_counts[group]:5d} train | "
            f"{val_counts[group]:5d} validation"
        )


def _fit_and_report_baselines(
    dataset: ExpressionDataset,
    tissues: list[str],
    groups: list[str],
    train_indices: list[int],
    val_indices: list[int],
    active_tissue: str,
) -> float:
    train_targets = np.asarray(
        [dataset.get_raw_target(index) for index in train_indices],
        dtype=np.float64,
    )
    val_targets = np.asarray(
        [dataset.get_raw_target(index) for index in val_indices],
        dtype=np.float64,
    )
    train_tissues = [tissues[index] for index in train_indices]
    val_tissues = [tissues[index] for index in val_indices]
    val_groups = [groups[index] for index in val_indices]

    constant = ConstantMeanBaseline().fit(train_targets)
    constant_predictions = constant.predict(len(val_indices))
    constant_metrics = compute_regression_metrics(
        constant_predictions, val_targets
    )

    tissue_only = TissueMeanBaseline().fit(train_tissues, train_targets)
    tissue_predictions = tissue_only.predict(val_tissues)
    tissue_metrics = compute_regression_metrics(tissue_predictions, val_targets)
    tissue_group_metrics = compute_group_metrics(
        tissue_predictions, val_targets, val_groups
    )

    print("Validation baselines fitted on the training split (raw target scale):")
    print(
        "  constant    | "
        f"mse={constant_metrics.mse:.6f}, "
        f"rmse={constant_metrics.rmse:.6f}, "
        f"mae={constant_metrics.mae:.6f}"
    )
    print(
        "  tissue-only | "
        f"mse={tissue_metrics.mse:.6f}, "
        f"rmse={tissue_metrics.rmse:.6f}, "
        f"mae={tissue_metrics.mae:.6f}"
    )
    print("  tissue-only groups:")
    print(format_group_metrics(tissue_group_metrics, active_tissue, indent="    "))
    return tissue_metrics.mse


def main() -> None:
    args = parse_arguments()
    config = _build_config(args)
    set_global_seed(args.seed)

    tee = TeeStream(args.log_file)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        device = resolve_device(config["device"])
        dataframe = load_expression_dataframe(args.data).reset_index(drop=True)
        tissues_present = {
            tissue.upper() for tissue in dataframe["tissue"].astype(str).unique()
        }
        if args.active_tissue.upper() not in tissues_present:
            raise ValueError(
                f"Active tissue {args.active_tissue!r} is absent from the dataset."
            )

        sorted_tissues = extract_unique_tissues(dataframe)
        vocabulary = build_vocabulary(sorted_tissues, config)
        dataset = ExpressionDataset(
            dataframe, vocabulary, config, sorted_tissues
        )

        data = build_diagnostic_dataloaders(
            dataset=dataset,
            dataframe=dataframe,
            config=config,
            pad_id=vocabulary.pad_id,
            active_tissue=args.active_tissue,
            motif=args.motif,
            seed=args.seed,
            balanced_sampler=args.balanced_sampler,
            samples_per_epoch=args.samples_per_epoch,
        )

        print(f"Device: {device}")
        print(f"Samples: {len(dataset)}")
        print(
            f"Train / validation: {len(data.train_indices)} / "
            f"{len(data.val_indices)}"
        )
        print(
            f"Rule under test: tissue={args.active_tissue!r}, "
            f"motif={args.motif!r}"
        )
        print(f"Seed: {args.seed}")
        if args.balanced_sampler:
            draws = args.samples_per_epoch or len(data.train_indices)
            present_groups = len(
                Counter(
                    data.group_labels[index] for index in data.train_indices
                )
            )
            print(
                "Training sampler: inverse-frequency WeightedRandomSampler "
                f"with replacement, {draws} draws/epoch; expected share "
                f"approximately {100.0 / present_groups:.1f}% per present group."
            )
        else:
            print("Training sampler: ordinary shuffled training split.")

        _print_group_distribution(
            data.group_labels,
            data.train_indices,
            data.val_indices,
            args.active_tissue,
        )
        tissue_baseline_mse = _fit_and_report_baselines(
            dataset=dataset,
            tissues=data.tissues,
            groups=data.group_labels,
            train_indices=data.train_indices,
            val_indices=data.val_indices,
            active_tissue=args.active_tissue,
        )

        model = ExpressionTransformer(
            vocabulary.vocab_size, vocabulary.pad_id, config
        ).to(device)
        callback = BestCheckpointEarlyStopping(
            checkpoint_path=args.checkpoint_file,
            patience=args.patience,
            min_delta=args.min_delta,
        )

        train_model_with_diagnostics(
            model=model,
            data=data,
            config=config,
            device=device,
            active_tissue=args.active_tissue,
            motif=args.motif,
            tissue_baseline_raw_mse=tissue_baseline_mse,
            callback=callback,
            max_grad_norm=args.max_grad_norm,
        )

        Path(args.normalizer_file).parent.mkdir(parents=True, exist_ok=True)
        data.normalizer.save(args.normalizer_file)
        print(f"Saved target normalizer: {args.normalizer_file}")
    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
