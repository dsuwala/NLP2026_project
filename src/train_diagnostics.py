#!/usr/bin/env python3
"""CLI for leakage-safe diagnostic training of the no-CNN Transformer."""
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
    DiagnosticDataBundle,
    build_diagnostic_dataloaders,
    set_global_seed,
    train_model_with_diagnostics,
)
from vocabulary import build_vocabulary


class TeeStream:
    """Duplicate stdout writes to the terminal and a log file."""

    def __init__(self, log_file_path: str) -> None:
        path = Path(log_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._terminal = sys.stdout
        self._log = path.open("w", encoding="utf-8")

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
        description=(
            "Train the no-CNN expression Transformer with gene-level splitting "
            "and truncation-aware four-group diagnostics."
        )
    )
    parser.add_argument("--data", required=True, help="Input CSV/TSV dataset.")
    parser.add_argument(
        "--active-tissue",
        default="HEAD",
        help="Active tissue used by the four-group diagnostic.",
    )
    parser.add_argument(
        "--motif",
        default="GATTACAA",
        help="DNA motif used by the four-group diagnostic.",
    )
    parser.add_argument(
        "--split-group-column",
        default="gene_id",
        help=(
            "Column kept intact across train/validation splits. For integrated "
            "data this should remain gene_id."
        ),
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
        default="logs/diagnostic_training.log",
        help="Path for console logs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--balanced-sampler",
        action="store_true",
        help=(
            "Use inverse-frequency row-group sampling on training only. This "
            "is generally not recommended for the primary real-data run."
        ),
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Sampler draws per epoch; default is the training-row count.",
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--start-lr", type=float, default=None)
    parser.add_argument("--max-lr", type=float, default=None)
    parser.add_argument("--warmup-period", type=int, default=None)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Legacy fixed-LR override. Sets start_lr=max_lr and disables "
            "warmup. Do not combine with the LR schedule arguments."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="AdamW weight decay; default is 0.01.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override CONFIG device, for example cuda, cuda:0 or cpu.",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> dict:
    """Copy project CONFIG and apply explicit command-line overrides."""

    config = copy.deepcopy(PROJECT_CONFIG)

    if args.lr is not None and any(
        value is not None
        for value in (args.start_lr, args.max_lr, args.warmup_period)
    ):
        raise ValueError(
            "--lr cannot be combined with --start-lr, --max-lr or "
            "--warmup-period."
        )

    direct_overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "device": args.device,
    }
    for key, value in direct_overrides.items():
        if value is not None:
            config[key] = value

    if args.lr is not None:
        config["start_lr"] = args.lr
        config["max_lr"] = args.lr
        config["warmup_period"] = 1
    else:
        if args.start_lr is not None:
            config["start_lr"] = args.start_lr
        if args.max_lr is not None:
            config["max_lr"] = args.max_lr
        if args.warmup_period is not None:
            config["warmup_period"] = args.warmup_period

    if args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay
    else:
        config.setdefault("weight_decay", 0.01)

    integer_minimums = {
        "epochs": 1,
        "batch_size": 1,
        "max_seq_len": 2,
        "gradient_accumulation_steps": 1,
        "warmup_period": 1,
    }
    for key, minimum in integer_minimums.items():
        if int(config[key]) < minimum:
            raise ValueError(f"{key} must be >= {minimum}.")

    if float(config["start_lr"]) <= 0 or float(config["max_lr"]) <= 0:
        raise ValueError("Learning rates must be positive.")
    if float(config["weight_decay"]) < 0:
        raise ValueError("weight_decay must be non-negative.")
    if args.patience < 1:
        raise ValueError("patience must be >= 1.")
    if args.min_delta < 0:
        raise ValueError("min_delta must be non-negative.")
    if args.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be non-negative.")
    if args.samples_per_epoch is not None and args.samples_per_epoch < 1:
        raise ValueError("samples_per_epoch must be positive.")

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

    print("Diagnostic groups based on the visible, truncated sequence:")
    for group in GROUP_ORDER:
        print(
            f"  {display_group_name(group, active_tissue):<30} "
            f"{total_counts[group]:7d} total | "
            f"{train_counts[group]:7d} train | "
            f"{val_counts[group]:7d} validation"
        )


def _print_data_diagnostics(
    data: DiagnosticDataBundle,
    total_samples: int,
    motif: str,
) -> None:
    train_groups = set(data.train_split_groups)
    val_groups = set(data.val_split_groups)
    overlap = train_groups & val_groups

    print(
        f"Grouped split column: {data.split_group_column!r} | "
        f"{len(train_groups)} train groups / {len(val_groups)} validation groups"
    )
    print(f"Split-group overlap: {len(overlap)}")
    if overlap:
        raise RuntimeError(
            "Train/validation leakage detected after grouped splitting."
        )

    print(
        "Input window: "
        f"max visible raw DNA={data.max_visible_dna_len} nt"
    )
    truncated_share = (
        100.0 * data.truncated_record_count / total_samples
        if total_samples
        else 0.0
    )
    print(
        "Truncation: "
        f"{data.truncated_record_count}/{total_samples} records "
        f"({truncated_share:.2f}%)"
    )
    print(
        f"Motif {motif!r}: full={data.full_motif_count}, "
        f"visible={data.visible_motif_count}, "
        "lost through truncation="
        f"{data.motif_lost_to_truncation_count}"
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
        constant_predictions,
        val_targets,
    )

    tissue_only = TissueMeanBaseline().fit(train_tissues, train_targets)
    tissue_predictions = tissue_only.predict(val_tissues)
    tissue_metrics = compute_regression_metrics(
        tissue_predictions,
        val_targets,
    )
    tissue_group_metrics = compute_group_metrics(
        tissue_predictions,
        val_targets,
        val_groups,
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
    print(
        format_group_metrics(
            tissue_group_metrics,
            active_tissue,
            indent="    ",
        )
    )
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

        if args.split_group_column not in dataframe.columns:
            raise ValueError(
                f"Split group column {args.split_group_column!r} is absent. "
                "Use a column such as gene_id that identifies repeated rows "
                "sharing the same genomic sequence."
            )

        tissues_present = {
            tissue.upper()
            for tissue in dataframe["tissue"].astype(str).unique()
        }
        if args.active_tissue.upper() not in tissues_present:
            raise ValueError(
                f"Active tissue {args.active_tissue!r} is absent from the dataset."
            )

        sorted_tissues = extract_unique_tissues(dataframe)
        vocabulary = build_vocabulary(sorted_tissues, config)
        dataset = ExpressionDataset(
            dataframe,
            vocabulary,
            config,
            sorted_tissues,
        )

        data = build_diagnostic_dataloaders(
            dataset=dataset,
            dataframe=dataframe,
            config=config,
            pad_id=vocabulary.pad_id,
            active_tissue=args.active_tissue,
            motif=args.motif,
            seed=args.seed,
            split_group_column=args.split_group_column,
            balanced_sampler=args.balanced_sampler,
            samples_per_epoch=args.samples_per_epoch,
        )

        print(f"Device: {device}")
        print(f"Samples: {len(dataset)}")
        print(f"Tokenization: {config.get('tokenization', 'nucleotide')}")
        if config.get("tokenization", "nucleotide") == "kmer":
            print(
                f"k-mer size / stride: {config['kmer_size']} / "
                f"{config['kmer_stride']}"
            )
        print(f"Vocabulary size: {vocabulary.vocab_size}")
        print(
            f"Train / validation rows: {len(data.train_indices)} / "
            f"{len(data.val_indices)}"
        )
        print(
            f"Diagnostic: tissue={args.active_tissue!r}, "
            f"motif={args.motif!r}"
        )
        print(f"Seed: {args.seed}")
        print(
            "Model interface: no CNN/MaxPool; dataset prepends the tissue token "
            "and model receives (tokens, key_padding_mask)."
        )
        _print_data_diagnostics(data, len(dataset), args.motif)

        print(
            "LR schedule: "
            f"start={float(config['start_lr']):.2e}, "
            f"max={float(config['max_lr']):.2e}, "
            f"warmup_period={int(config['warmup_period'])}"
        )
        print(
            "Batching: "
            f"micro_batch={int(config['batch_size'])}, "
            "gradient_accumulation_steps="
            f"{int(config['gradient_accumulation_steps'])}, "
            "effective_batch="
            f"{int(config['batch_size']) * int(config['gradient_accumulation_steps'])}"
        )
        print(f"AdamW weight decay: {float(config['weight_decay']):.6g}")

        if args.balanced_sampler:
            present_groups = len(
                Counter(
                    data.group_labels[index]
                    for index in data.train_indices
                )
            )
            draws = args.samples_per_epoch or len(data.train_indices)
            print(
                "Training sampler: inverse-frequency diagnostic-row sampling "
                f"with replacement, {draws} draws/epoch; expected share "
                f"approximately {100.0 / present_groups:.1f}% per present group."
            )
            print(
                "Warning: balanced diagnostic sampling changes the empirical "
                "real-data distribution and should not be the primary run."
            )
        else:
            print("Training sampler: ordinary shuffled training rows.")

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
            vocabulary.vocab_size,
            vocabulary.pad_id,
            config,
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
            max_grad_norm=(
                args.max_grad_norm
                if args.max_grad_norm > 0
                else None
            ),
        )

        Path(args.normalizer_file).parent.mkdir(parents=True, exist_ok=True)
        data.normalizer.save(args.normalizer_file)
        print(f"Saved target normalizer: {args.normalizer_file}")
    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
