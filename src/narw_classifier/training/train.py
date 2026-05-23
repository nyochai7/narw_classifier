"""Hydra-driven training entrypoint for the EfficientNet baseline.

Run::

    uv run python -m narw_classifier.training.train

Override any field via Hydra CLI, e.g.::

    uv run python -m narw_classifier.training.train trainer.max_epochs=20 model.lr=3e-4
"""

from __future__ import annotations

import logging
from datetime import datetime

import hydra
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from ..data.datamodule import NARWDataModule
from ..models.baseline import BaselineEfficientNet
from ..models.preprocess import MelImagePreprocessor
from ..utils.lightning import build_checkpoint_callback, build_logger, format_lr

log = logging.getLogger(__name__)


def build_preprocessor(cfg: DictConfig) -> MelImagePreprocessor:
    return MelImagePreprocessor(
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


def build_datamodule(cfg: DictConfig) -> NARWDataModule:
    # Hydra changes CWD to outputs/<date>/<time>/ — resolve `data.root` against
    # the original CWD so relative paths in conf/data/*.yaml keep working.
    return NARWDataModule(
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


def build_model(cfg: DictConfig, preprocessor: MelImagePreprocessor) -> BaselineEfficientNet:
    return BaselineEfficientNet(
        preprocessor=preprocessor,
        freeze_mode=cfg.model.freeze_mode,
        pretrained=cfg.model.pretrained,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        scheduler=cfg.model.scheduler,
        max_epochs=cfg.trainer.max_epochs,
    )


def default_run_name(cfg: DictConfig, now: datetime | None = None) -> str:
    """W&B run name: ``{freeze_mode}_lr{lr}_bs{bs}_e{epochs}_{HHMMSS}``."""
    ts = (now or datetime.now()).strftime("%H%M%S")
    return (
        f"{cfg.model.freeze_mode}"
        f"_lr{format_lr(cfg.model.lr)}"
        f"_bs{cfg.data.batch_size}"
        f"_e{cfg.trainer.max_epochs}"
        f"_{ts}"
    )


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    pl.seed_everything(cfg.seed, workers=True)

    preprocessor = build_preprocessor(cfg)
    datamodule = build_datamodule(cfg)
    model = build_model(cfg, preprocessor)

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
