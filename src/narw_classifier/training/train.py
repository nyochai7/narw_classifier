"""Hydra-driven training entrypoint.

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
    # Hydra changes CWD to outputs/.../ — resolve `data.root` against the original CWD
    # so relative paths in conf/data/*.yaml still work.
    return NARWDataModule(
        data_root=to_absolute_path(cfg.data.root),
        train_subdir=cfg.data.train_subdir,
        target_sample_rate=cfg.preprocess.sample_rate,
        target_duration_s=cfg.data.target_duration_s,
        val_fraction=cfg.data.val_fraction,
        split_seed=cfg.data.split_seed,
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


def _format_lr(lr: float) -> str:
    """Compact LR string: 1e-3, 3.5e-4 (drops trailing zeros + leading exp zero)."""
    mantissa, exp = f"{lr:.2e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def default_run_name(cfg: DictConfig, now: datetime | None = None) -> str:
    """Build an informative W&B run name from the resolved cfg.

    Format: ``{freeze_mode}_lr{lr}_bs{batch_size}_e{epochs}_{HHMMSS}``
    Example: ``linear_probe_lr1e-3_bs32_e1_104242``
    """
    ts = (now or datetime.now()).strftime("%H%M%S")
    return (
        f"{cfg.model.freeze_mode}"
        f"_lr{_format_lr(cfg.model.lr)}"
        f"_bs{cfg.data.batch_size}"
        f"_e{cfg.trainer.max_epochs}"
        f"_{ts}"
    )


def build_logger(cfg: DictConfig):
    if cfg.logger.kind == "none":
        return False
    if cfg.logger.kind == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        name = cfg.logger.run_name or default_run_name(cfg)
        return WandbLogger(
            project=cfg.logger.project,
            name=name,
            tags=list(cfg.logger.tags) if cfg.logger.tags else None,
            save_dir=cfg.logger.save_dir,
            offline=cfg.logger.offline,
        )
    raise ValueError(f"Unknown logger.kind: {cfg.logger.kind}")


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    pl.seed_everything(cfg.seed, workers=True)

    preprocessor = build_preprocessor(cfg)
    datamodule = build_datamodule(cfg)
    model = build_model(cfg, preprocessor)
    logger = build_logger(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        fast_dev_run=cfg.trainer.fast_dev_run,
        deterministic=cfg.trainer.deterministic,
        logger=logger,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
