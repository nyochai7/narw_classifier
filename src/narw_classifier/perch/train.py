"""Train the linear probe on cached Perch embeddings.

Run (after extracting embeddings)::

    uv run python -m narw_classifier.perch.train

Continues a previous run via ``ckpt_path=...`` Hydra override.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from .datamodule import PerchEmbeddingsDataModule
from .linear_probe import PerchLinearProbe

log = logging.getLogger(__name__)


def _format_lr(lr: float) -> str:
    mantissa, exp = f"{lr:.2e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def default_run_name(cfg: DictConfig, now: datetime | None = None) -> str:
    ts = (now or datetime.now()).strftime("%H%M%S")
    return (
        f"perch_v2_linear"
        f"_lr{_format_lr(cfg.model.lr)}"
        f"_bs{cfg.data.batch_size}"
        f"_e{cfg.trainer.max_epochs}"
        f"_{ts}"
    )


def build_callbacks(cfg: DictConfig) -> list:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    return [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best",
            monitor="val/auroc",
            mode="max",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]


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


@hydra.main(version_base=None, config_path="../../../conf", config_name="perch_config")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    pl.seed_everything(cfg.seed, workers=True)

    cache_dir = Path(to_absolute_path(cfg.embeddings.cache_dir)) / cfg.data.split_strategy
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

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        fast_dev_run=cfg.trainer.fast_dev_run,
        deterministic=cfg.trainer.deterministic,
        callbacks=build_callbacks(cfg),
        logger=build_logger(cfg),
    )
    ckpt_path = to_absolute_path(cfg.ckpt_path) if cfg.get("ckpt_path") else None
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
