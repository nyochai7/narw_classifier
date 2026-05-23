"""Single Hydra-driven training entrypoint for both pipelines.

Dispatches on ``cfg.pipeline``:
- ``efficientnet``: log-mel spectrogram → fine-tuned EfficientNet-B3
- ``perch``: linear probe on cached Perch v2 embeddings

Run the EfficientNet baseline (default)::

    uv run python -m narw_classifier.training.train

Run the Perch linear probe::

    uv run python -m narw_classifier.training.train --config-name=perch_config

Override any field via Hydra CLI, e.g.::

    uv run python -m narw_classifier.training.train trainer.max_epochs=20 model.lr=3e-4

Resume a run by pointing ``ckpt_path`` at a saved checkpoint::

    uv run python -m narw_classifier.training.train \\
        ckpt_path=outputs/2026-05-23/10-42-17/checkpoints/last.ckpt
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from ..analysis.callback import EvalArtifactsCallback
from ..data.datamodule import NARWDataModule
from ..data.perch_datamodule import PerchEmbeddingsDataModule
from ..models.baseline import BaselineEfficientNet
from ..models.perch_linear_probe import PerchLinearProbe
from ..models.preprocess import MelImagePreprocessor
from ..utils.lightning import build_checkpoint_callback, build_logger, format_lr

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EfficientNet baseline: log-mel spectrogram → ImageNet-pretrained CNN
# ---------------------------------------------------------------------------


def _build_efficientnet(cfg: DictConfig) -> tuple[pl.LightningDataModule, pl.LightningModule]:
    preprocessor = MelImagePreprocessor(
        sample_rate=cfg.preprocess.sample_rate,
        n_fft=cfg.preprocess.n_fft,
        win_length=cfg.preprocess.win_length,
        hop_length=cfg.preprocess.hop_length,
        n_mels=cfg.preprocess.n_mels,
        f_min=cfg.preprocess.f_min,
        f_max=cfg.preprocess.f_max,
        image_size=cfg.preprocess.image_size,
        top_db=cfg.preprocess.top_db,
    )
    datamodule = NARWDataModule(
        data_root=to_absolute_path(cfg.data.root),
        train_subdir=cfg.data.train_subdir,
        target_sample_rate=cfg.preprocess.sample_rate,
        target_duration_s=cfg.data.target_duration_s,
        val_fraction=cfg.data.val_fraction,
        split_seed=cfg.data.split_seed,
        split_strategy=cfg.data.split_strategy,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        balanced_train_sampler=cfg.data.balanced_train_sampler,
    )
    model = BaselineEfficientNet(
        preprocessor=preprocessor,
        freeze_mode=cfg.model.freeze_mode,
        pretrained=cfg.model.pretrained,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        scheduler=cfg.model.scheduler,
        max_epochs=cfg.trainer.max_epochs,
    )
    return datamodule, model


def _efficientnet_run_name(cfg: DictConfig, ts: str) -> str:
    return (
        f"{cfg.model.freeze_mode}"
        f"_lr{format_lr(cfg.model.lr)}"
        f"_bs{cfg.data.batch_size}"
        f"_e{cfg.trainer.max_epochs}"
        f"_{ts}"
    )


# ---------------------------------------------------------------------------
# Perch linear probe: pre-extracted embeddings → Linear(1536, 1)
# ---------------------------------------------------------------------------


def _build_perch(cfg: DictConfig) -> tuple[pl.LightningDataModule, pl.LightningModule]:
    shift_tag = f"pitch_shift_{int(cfg.preprocess.pitch_shift_semitones)}"
    cache_dir = (
        Path(to_absolute_path(cfg.embeddings.cache_dir)) / cfg.data.split_strategy / shift_tag
    )
    datamodule = PerchEmbeddingsDataModule(
        cache_dir=cache_dir,
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
    return datamodule, model


def _perch_run_name(cfg: DictConfig, ts: str) -> str:
    shift = int(cfg.preprocess.pitch_shift_semitones)
    shift_tag = f"shift{shift}" if shift else "noshift"
    return (
        f"perch_v2_linear_{shift_tag}"
        f"_lr{format_lr(cfg.model.lr)}"
        f"_bs{cfg.data.batch_size}"
        f"_e{cfg.trainer.max_epochs}"
        f"_{ts}"
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------


_PIPELINES = {
    "efficientnet": (_build_efficientnet, _efficientnet_run_name),
    "perch": (_build_perch, _perch_run_name),
}


def default_run_name(cfg: DictConfig, now: datetime | None = None) -> str:
    ts = (now or datetime.now()).strftime("%H%M%S")
    _, name_fn = _PIPELINES[cfg.pipeline]
    return name_fn(cfg, ts)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    pl.seed_everything(cfg.seed, workers=True)

    if cfg.pipeline not in _PIPELINES:
        raise ValueError(
            f"Unknown pipeline: {cfg.pipeline!r} (expected one of {sorted(_PIPELINES)!r})"
        )
    build_fn, _ = _PIPELINES[cfg.pipeline]
    datamodule, model = build_fn(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        fast_dev_run=cfg.trainer.fast_dev_run,
        deterministic=cfg.trainer.deterministic,
        callbacks=[build_checkpoint_callback(), EvalArtifactsCallback()],
        logger=build_logger(cfg, default_run_name=default_run_name(cfg)),
    )
    ckpt_path = to_absolute_path(cfg.ckpt_path) if cfg.get("ckpt_path") else None
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
