# NLP2026_project
This repository contains project for Natural Language Processing course conducted at MIMUW in the summer semester 2025/26

To run the diagnostic training pipeline on the synthetic dataset:

PYTHONPATH=src python src/train_diagnostics.py \
  --data synthetic_data/1_small.tsv \
  --active-tissue HEAD \
  --motif GATTACAA \
  --epochs 100 \
  --patience 10 \
  --seed 42 \
  --log-file logs/diagnostics_seed_42.log \
  --normalizer-file checkpoints/normalizer_seed_42.json \
  --checkpoint-file checkpoints/model_seed_42.pt

