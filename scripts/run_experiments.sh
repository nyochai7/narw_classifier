#!/usr/bin/env bash
# Run the three baseline experiments end-to-end.
#
# Assumes you have already:
#   - cloned the repo and run `uv sync`
#   - placed the dataset at data/raw/{train2, test2, sampleSubmission.csv}
#   - run `uv run wandb login`
#
# Each experiment logs to W&B project `narw-classifier` with a self-describing
# run name. Run from the repo root.

set -euo pipefail

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

# -----------------------------------------------------------------------------
step "Experiment 1 — EfficientNet-B3 linear probe (head only), 10 epochs"
uv run python -m narw_classifier.training.train \
    model=efficientnet_b3_linear_probe \
    trainer.max_epochs=10

# -----------------------------------------------------------------------------
step "Experiment 2 — Perch v2 linear probe, NO pitch shift"

step "  2a) extract embeddings (pitch_shift_semitones=0)"
uv run python -m narw_classifier.training.extract_perch_embeddings \
    preprocess.pitch_shift_semitones=0

step "  2b) train linear probe on cached embeddings"
uv run python -m narw_classifier.training.train \
    --config-name=perch_config \
    preprocess.pitch_shift_semitones=0 \
    trainer.max_epochs=10

# -----------------------------------------------------------------------------
step "Experiment 3 — Perch v2 linear probe, pitch_shift_semitones=36"

step "  3a) extract embeddings (pitch_shift_semitones=36, default)"
uv run python -m narw_classifier.training.extract_perch_embeddings

step "  3b) train linear probe on cached embeddings"
uv run python -m narw_classifier.training.train \
    --config-name=perch_config \
    trainer.max_epochs=10

step "All three experiments complete. Compare runs in W&B project: narw-classifier"
