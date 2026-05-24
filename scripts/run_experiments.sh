#!/usr/bin/env bash
# Run the four baseline experiments end-to-end.
#
# Assumes you have already:
#   - cloned the repo and run `uv sync`
#   - placed the dataset at data/raw/{train2, test2, sampleSubmission.csv}
#   - run `uv run wandb login`
#
# Each experiment logs to W&B project `narw-classifier` with a self-describing
# run name. Run from the repo root.
#
# Notes:
#   - The Perch extract step is one-time per pitch_shift value; subsequent
#     runs reuse the cached embeddings.
#   - Defaults to `trainer.precision=16-mixed` (CUDA). On Mac override with
#     `trainer.precision=32 trainer.accelerator=cpu` per command.

set -euo pipefail

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

# -----------------------------------------------------------------------------
step "Experiment 1 — EfficientNet-B3 linear probe (head only), 10 epochs"
uv run python -m narw_classifier.training.train \
    model=efficientnet_b3_linear_probe \
    trainer.max_epochs=10

# -----------------------------------------------------------------------------
step "Experiment 2 — EfficientNet-B3 full fine-tune, 10 epochs"
uv run python -m narw_classifier.training.train \
    model=efficientnet_b3_full_finetune \
    trainer.max_epochs=10

# -----------------------------------------------------------------------------
step "Experiment 3 — Perch v2 linear probe, NO pitch shift, 100 epochs"

step "  3a) extract embeddings (pitch_shift_semitones=0)"
uv run python -m narw_classifier.training.extract_perch_embeddings \
    preprocess.pitch_shift_semitones=0

step "  3b) train linear probe on cached embeddings"
uv run python -m narw_classifier.training.train \
    --config-name=perch_config \
    preprocess.pitch_shift_semitones=0 \
    trainer.max_epochs=100

# -----------------------------------------------------------------------------
step "Experiment 4 — Perch v2 linear probe, +36 semitone pitch shift, 100 epochs"

step "  4a) extract embeddings (pitch_shift_semitones=36, default)"
uv run python -m narw_classifier.training.extract_perch_embeddings

step "  4b) train linear probe on cached embeddings"
uv run python -m narw_classifier.training.train \
    --config-name=perch_config \
    trainer.max_epochs=100

step "All four experiments complete. Compare runs in W&B project: narw-classifier"
