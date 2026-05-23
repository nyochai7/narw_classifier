"""Lightning Trainer plumbing shared across the EfficientNet baseline and Perch
linear-probe training entrypoints: run-name formatting, checkpoint + logger wiring."""

from __future__ import annotations

from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def format_lr(lr: float) -> str:
    """Compact LR string: ``1e-3``, ``3.5e-4`` (drops trailing zeros + leading exp zero)."""
    mantissa, exp = f"{lr:.2e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def build_checkpoint_callback() -> ModelCheckpoint:
    """Save ``best.ckpt`` (highest ``val/auroc``) + ``last.ckpt`` (for resume) under the
    Hydra run dir (``outputs/<date>/<time>/checkpoints/``)."""
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    return ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best",
        monitor="val/auroc",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )


def build_logger(cfg: DictConfig, default_run_name: str):
    """Return a Lightning logger from ``cfg.logger`` (``wandb`` or ``none``).

    ``default_run_name`` is used when ``cfg.logger.run_name`` is null.
    """
    if cfg.logger.kind == "none":
        return False
    if cfg.logger.kind == "wandb":
        name = cfg.logger.run_name or default_run_name
        return WandbLogger(
            project=cfg.logger.project,
            name=name,
            tags=list(cfg.logger.tags) if cfg.logger.tags else None,
            save_dir=cfg.logger.save_dir,
            offline=cfg.logger.offline,
        )
    raise ValueError(f"Unknown logger.kind: {cfg.logger.kind}")
