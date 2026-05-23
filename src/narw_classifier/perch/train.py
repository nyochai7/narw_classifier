"""Hydra-driven training entrypoint for the Perch v2 linear probe.

Run (after extracting embeddings via ``narw_classifier.perch.extract``)::

    uv run python -m narw_classifier.perch.train

Continue a previous run via ``ckpt_path=...``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from ..utils.lightning import build_checkpoint_callback, build_logger, format_lr
from .datamodule import PerchEmbeddingsDataModule
from .linear_probe import PerchLinearProbe

log = logging.getLogger(__name__)


def _cache_dir(cfg: DictConfig) -> Path:
    """Resolve the embedding cache subdir for the current ``(split, pitch_shift)``."""
    shift_tag = f"pitch_shift_{int(cfg.preprocess.pitch_shift_semitones)}"
    return Path(to_absolute_path(cfg.embeddings.cache_dir)) / cfg.data.split_strategy / shift_tag


def default_run_name(cfg: DictConfig, now: datetime | None = None) -> str:
    """W&B run name: ``perch_v2_linear_{shift_tag}_lr{lr}_bs{bs}_e{epochs}_{HHMMSS}``."""
    ts = (now or datetime.now()).strftime("%H%M%S")
    shift = int(cfg.preprocess.pitch_shift_semitones)
    shift_tag = f"shift{shift}" if shift else "noshift"
    return (
        f"perch_v2_linear_{shift_tag}"
        f"_lr{format_lr(cfg.model.lr)}"
        f"_bs{cfg.data.batch_size}"
        f"_e{cfg.trainer.max_epochs}"
        f"_{ts}"
    )


@hydra.main(version_base=None, config_path="../../../conf", config_name="perch_config")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    pl.seed_everything(cfg.seed, workers=True)

    datamodule = PerchEmbeddingsDataModule(
        cache_dir=_cache_dir(cfg),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        balanced_train_sampler=cfg.data.balanced_train_sampler,
        seed=cfg.data.split_seed,
    )
    model = PerchLinearProbe(
        embedding_dim=cfg.model.embedding_dim,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        fast_dev_run=cfg.trainer.fast_dev_run,
        deterministic=cfg.trainer.deterministic,
        callbacks=[build_checkpoint_callback()],
        logger=build_logger(cfg, default_run_name=default_run_name(cfg)),
    )
    ckpt_path = to_absolute_path(cfg.ckpt_path) if cfg.get("ckpt_path") else None
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
