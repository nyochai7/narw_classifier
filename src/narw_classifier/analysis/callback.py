"""Lightning callback that produces FP/FN analysis artifacts after training.

At end of training, runs inference on the val and test sets and saves:

    outputs/<date>/<time>/analysis/
        val/  { predictions.npz, pr_curve.png, roc_curve.png,
                confusion_matrix_t0.5.png, summary.json }
        test/ { ... same files ... }

If a WandbLogger is attached, the figures + summary scalars are uploaded under
``final/val/*`` and ``final/test/*`` so the run dashboard has everything for
the comparison report without re-running anything.
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


def _save_and_log_split(
    *,
    split_name: str,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    loader,
    files: list[str] | None,
    base_out_dir: Path,
) -> None:
    if loader is None:
        log.warning("EvalArtifactsCallback: no %s dataloader; skipping.", split_name)
        return
    probs, labels = compute_predictions(pl_module, loader)
    if files is not None and len(files) != len(probs):
        log.warning(
            "EvalArtifactsCallback[%s]: files (%d) != predictions (%d); dropping filenames.",
            split_name,
            len(files),
            len(probs),
        )
        files = None

    out_dir = base_out_dir / split_name
    summary = save_run_artifacts(out_dir, probs, labels, files)
    log.info("EvalArtifactsCallback[%s]: %s", split_name, out_dir)
    log.info("EvalArtifactsCallback[%s]: summary = %s", split_name, summary)

    if isinstance(trainer.logger, WandbLogger):
        import wandb

        run = trainer.logger.experiment
        run.summary.update({f"final/{split_name}/{k}": v for k, v in summary.items()})
        run.log(
            {
                f"final/{split_name}/pr_curve": wandb.Image(str(out_dir / "pr_curve.png")),
                f"final/{split_name}/roc_curve": wandb.Image(str(out_dir / "roc_curve.png")),
                f"final/{split_name}/confusion_matrix_t0.5": wandb.Image(
                    str(out_dir / "confusion_matrix_t0.5.png")
                ),
            }
        )


class EvalArtifactsCallback(pl.Callback):
    """At end of training, save analysis artifacts for both the val and test splits."""

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        dm = trainer.datamodule
        base_out_dir = Path(HydraConfig.get().runtime.output_dir) / "analysis"

        log.info("EvalArtifactsCallback: computing val predictions...")
        _save_and_log_split(
            split_name="val",
            trainer=trainer,
            pl_module=pl_module,
            loader=trainer.val_dataloaders,
            files=getattr(dm, "val_files", None) or None,
            base_out_dir=base_out_dir,
        )

        # Only try to compute test artifacts if the DataModule actually provides
        # a test loader (the LightningDataModule base class has a stub that raises).
        test_loader = None
        try:
            test_loader = dm.test_dataloader()
        except Exception as e:
            log.info("EvalArtifactsCallback: no test dataloader (%s); skipping test artifacts.", e)
        if test_loader is not None:
            log.info("EvalArtifactsCallback: computing test predictions...")
            _save_and_log_split(
                split_name="test",
                trainer=trainer,
                pl_module=pl_module,
                loader=test_loader,
                files=getattr(dm, "test_files", None) or None,
                base_out_dir=base_out_dir,
            )
