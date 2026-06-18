#!/usr/bin/env python3
"""
CLI driver for training the genomic expression Transformer.

Exposes CONFIG hyperparameters at module scope and orchestrates data loading,
model construction, and the training loop with tee'd logging to file.
"""

from __future__ import annotations

import argparse
import sys

from data_utils import extract_unique_tissues, load_expression_dataframe
from dataset import ExpressionDataset
from model import ExpressionTransformer
from training import build_dataloaders, resolve_device, train_model
from vocabulary import build_vocabulary

CONFIG = {
    # Data params
    "vocab_pad": "[PAD]",
    "vocab_unk": "N",
    "nucleotides": ["A", "C", "T", "G"],
    "max_seq_len": 1500,
    # Model params
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "d_mlp": 512,
    "dropout": 0.1,
    # Training params
    "batch_size": 32,
    "lr": 1e-4,
    "epochs": 150,
    "val_split": 0.1,
    "device": "cuda",
}


class TeeStream:
    """Duplicate stdout writes to both the terminal and a log file."""

    def __init__(self, log_file_path: str) -> None:
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
    """Parse command-line arguments for the training driver.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train genomic expression Transformer."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the dataset CSV/TSV.",
    )
    parser.add_argument(
        "--log-file",
        default="training_logs.txt",
        help="Path to output training logs (default: training_logs.txt).",
    )
    parser.add_argument(
        "--normalizer-file",
        default="target_normalizer.json",
        help="Path to save fitted target normalizer for inference.",
    )
    return parser.parse_args()


def main() -> None:
    """Load data, build model, and run the training loop."""
    args = parse_arguments()
    tee = TeeStream(args.log_file)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        device = resolve_device(CONFIG["device"])

        df = load_expression_dataframe(args.data)
        sorted_tissues = extract_unique_tissues(df)
        vocabulary = build_vocabulary(sorted_tissues, CONFIG)

        dataset = ExpressionDataset(df, vocabulary, CONFIG, sorted_tissues)
        train_loader, val_loader, normalizer = build_dataloaders(
            dataset, CONFIG, vocabulary.pad_id
        )

        model = ExpressionTransformer(
            vocabulary.vocab_size, vocabulary.pad_id, CONFIG
        ).to(device)

        print(
            "Target normalizer (train split): "
            f"min={normalizer.min_val:.6f}, max={normalizer.max_val:.6f}, "
            f"mean={normalizer.mean:.6f}, std={normalizer.std:.6f}"
        )

        train_model(model, train_loader, val_loader, CONFIG, device)
        normalizer.save(args.normalizer_file)
        print(f"Saved target normalizer: {args.normalizer_file}")
    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
