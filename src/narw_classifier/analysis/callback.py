"""Lightning callback that produces FP/FN analysis artifacts at end of training.

For each finished run, saves under ``outputs/<date>/<time>/analysis/``:
    - predictions.npz   (probs, labels, val filenames)
    - pr_curve.png
    - roc_curve.png
    - confusion_matrix_t0.5.png
    - summary.json      (AUROC, AP, recall@1%FPR, recall@5%FPR, ...)

If a WandbLogger is attached, the figures + summary scalars are also uploaded
to the W&B run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.loggers import WandbLogger

from .artifacts import save_run_artifacts
from .predictions import compute_predictions

log = logging.getLogger(__name__)


class EvalArtifactsCallback(pl.Callback):
    """Computes + saves analysis artifacts from the val set at the end of training."""

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        val_loader = trainer.val_dataloaders
        if val_loader is None:
            log.warning("EvalArtifactsCallback: no val_dataloader; skipping.")
            return

        log.info("EvalArtifactsCallback: computing val predictions...")
        probs, labels = compute_predictions(pl_module, val_loader)
        files = getattr(trainer.datamodule, "val_files", None) or None
        if files is not None and len(files) != len(probs):
            log.warning(
                "EvalArtifactsCallback: val_files length (%d) != predictions (%d); "
                "dropping filenames.",
                len(files),
                len(probs),
            )
            files = None

        out_dir = Path(HydraConfig.get().runtime.output_dir) / "analysis"
        summary = save_run_artifacts(out_dir, probs, labels, files)
        log.info("EvalArtifactsCallback: artifacts saved to %s", out_dir)
        log.info("EvalArtifactsCallback: summary = %s", summary)

        if isinstance(trainer.logger, WandbLogger):
            import wandb

            run = trainer.logger.experiment
            run.summary.update({f"final/{k}": v for k, v in summary.items()})
            run.log(
                {
                    "final/pr_curve": wandb.Image(str(out_dir / "pr_curve.png")),
                    "final/roc_curve": wandb.Image(str(out_dir / "roc_curve.png")),
                    "final/confusion_matrix_t0.5": wandb.Image(
                        str(out_dir / "confusion_matrix_t0.5.png")
                    ),
                }
            )
