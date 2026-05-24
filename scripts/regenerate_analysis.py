"""Regenerate the analysis artifacts for one or more existing training runs.

Loads ``best.ckpt`` (or ``last.ckpt`` if best isn't there) from each run dir,
rebuilds the model + datamodule from the saved Hydra config, runs val inference,
and writes ``predictions.npz`` + ``pr_curve.png`` + ``roc_curve.png`` +
``confusion_matrix_t0.5.png`` + ``summary.json`` into ``RUN_DIR/analysis/``.

This is the same set of artifacts the ``EvalArtifactsCallback`` produces during
training; use this script to backfill analysis for runs that finished *before*
the callback was wired in.

Usage::

    uv run python scripts/regenerate_analysis.py RUN_DIR [RUN_DIR ...]

E.g.::

    uv run python scripts/regenerate_analysis.py \\
        outputs/2026-05-23/15-02-36_perch \\
        outputs/2026-05-22/17-29-12
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from pathlib import Path

import pytorch_lightning as pl  # noqa: F401  — needed for LightningModule subclasses
import torch.serialization
from omegaconf import DictConfig, OmegaConf

from narw_classifier.analysis.artifacts import save_run_artifacts
from narw_classifier.analysis.predictions import compute_predictions

# Our DataModules save `cache_dir` etc. as ``pathlib.PosixPath`` hyperparameters;
# PyTorch 2.6+ defaults torch.load to ``weights_only=True`` which rejects them.
# These checkpoints come from our own training runs, so allowlisting is safe.
torch.serialization.add_safe_globals(
    [pathlib.PosixPath, pathlib.WindowsPath, pathlib.PurePosixPath]
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("regenerate_analysis")


def _infer_pipeline(cfg: DictConfig) -> str:
    """Return ``efficientnet`` or ``perch`` from a saved config.

    Newer configs have an explicit ``pipeline:`` field; older ones don't,
    so we sniff the shape of ``cfg.model``.
    """
    if "pipeline" in cfg:
        return cfg.pipeline
    if "freeze_mode" in cfg.model:
        return "efficientnet"
    if "embedding_dim" in cfg.model:
        return "perch"
    raise ValueError(
        "Cannot infer pipeline: config has no `pipeline:` field and `cfg.model` "
        "doesn't look like either an EfficientNet or Perch config."
    )


def _build_efficientnet(cfg: DictConfig):
    from narw_classifier.data.datamodule import NARWDataModule
    from narw_classifier.models.baseline import BaselineEfficientNet
    from narw_classifier.models.preprocess import MelImagePreprocessor

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
        data_root=str(Path(cfg.data.root).expanduser().resolve()),
        train_subdir=cfg.data.train_subdir,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.get("test_fraction", 0.15),
        split_seed=cfg.data.split_seed,
        split_strategy=cfg.data.get("split_strategy", "day_stratified"),
        batch_size=cfg.data.batch_size,
        num_workers=0,  # don't fork extra workers for one-off inference
        pin_memory=False,
        balanced_train_sampler=False,
    )
    return preprocessor, datamodule, BaselineEfficientNet


def _build_perch(cfg: DictConfig):
    from narw_classifier.data.perch_datamodule import PerchEmbeddingsDataModule
    from narw_classifier.models.perch_linear_probe import PerchLinearProbe

    shift_tag = f"pitch_shift_{int(cfg.preprocess.pitch_shift_semitones)}"
    cache_dir = (
        Path(cfg.embeddings.cache_dir).expanduser().resolve() / cfg.data.split_strategy / shift_tag
    )
    datamodule = PerchEmbeddingsDataModule(
        cache_dir=cache_dir,
        batch_size=cfg.data.batch_size,
        num_workers=0,
        balanced_train_sampler=False,
        seed=cfg.data.split_seed,
    )
    return None, datamodule, PerchLinearProbe


def _pick_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    for name in ("best.ckpt", "last.ckpt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")


def regenerate(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No .hydra/config.yaml under {run_dir}")
    cfg = OmegaConf.load(cfg_path)
    pipeline = _infer_pipeline(cfg)
    log.info("[%s] pipeline=%s", run_dir.name, pipeline)

    ckpt_path = _pick_checkpoint(run_dir)
    log.info("[%s] checkpoint=%s", run_dir.name, ckpt_path.name)

    if pipeline == "efficientnet":
        preprocessor, datamodule, model_cls = _build_efficientnet(cfg)
        model = model_cls.load_from_checkpoint(ckpt_path, preprocessor=preprocessor)
    elif pipeline == "perch":
        _, datamodule, model_cls = _build_perch(cfg)
        model = model_cls.load_from_checkpoint(ckpt_path)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}")

    datamodule.prepare_data()
    datamodule.setup(stage="validate")
    probs, labels = compute_predictions(model, datamodule.val_dataloader())
    files = getattr(datamodule, "val_files", None) or None
    if files is not None and len(files) != len(probs):
        files = None

    out_dir = run_dir / "analysis"
    summary = save_run_artifacts(out_dir, probs, labels, files)
    log.info("[%s] wrote %s", run_dir.name, out_dir)
    log.info("[%s] summary = %s", run_dir.name, summary)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Hydra run dirs (each must contain .hydra/config.yaml and checkpoints/)",
    )
    args = parser.parse_args()

    n_failed = 0
    for rd in args.run_dirs:
        try:
            regenerate(rd)
        except Exception as e:
            log.error("[%s] FAILED: %s", rd, e)
            n_failed += 1
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
