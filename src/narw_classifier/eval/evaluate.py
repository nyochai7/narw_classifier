"""Hydra-driven evaluation entrypoint. Runs validation on the held-out split.

Run::

    uv run python -m narw_classifier.eval.evaluate ckpt_path=outputs/.../ckpt.ckpt
"""

from __future__ import annotations

import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from ..models.baseline import BaselineEfficientNet
from ..training.train import build_datamodule, build_logger, build_preprocessor

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    if not cfg.get("ckpt_path"):
        raise ValueError(
            "ckpt_path must be set, e.g. `ckpt_path=outputs/.../checkpoints/last.ckpt`"
        )

    pl.seed_everything(cfg.seed, workers=True)
    preprocessor = build_preprocessor(cfg)
    datamodule = build_datamodule(cfg)
    model = BaselineEfficientNet.load_from_checkpoint(cfg.ckpt_path, preprocessor=preprocessor)
    logger = build_logger(cfg)
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=logger,
    )
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
