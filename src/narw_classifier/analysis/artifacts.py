"""High-level 'produce all analysis artifacts for one split' helper.

Used by both the :class:`EvalArtifactsCallback` (in-training) and the
``02_error_analysis.ipynb`` notebook (post-hoc).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import best_f1_from_sweep, confusion_metrics_at_threshold, recall_at_fpr
from .plots import plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from .predictions import save_predictions


def compute_summary(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Headline numbers for the W&B summary table / report.

    Includes everything needed for FP / FN error analysis at multiple operating
    points (default 0.5, plus 1% / 5% FPR constraints, plus best-F1).
    """
    rec_1, thr_1 = recall_at_fpr(probs, labels, target_fpr=0.01)
    rec_5, thr_5 = recall_at_fpr(probs, labels, target_fpr=0.05)
    best_f1, best_f1_t = best_f1_from_sweep(probs, labels)
    cm = confusion_metrics_at_threshold(probs, labels, threshold=0.5)

    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "ap": float(average_precision_score(labels, probs)),
        "accuracy_t0.5": float(
            (cm["tp"] + cm["tn"]) / max(cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"], 1)
        ),
        "precision_t0.5": cm["precision"],
        "recall_t0.5": cm["recall"],
        "f1_t0.5": cm["f1"],
        "fpr_t0.5": cm["fpr"],
        "tp_t0.5": cm["tp"],
        "fp_t0.5": cm["fp"],
        "fn_t0.5": cm["fn"],
        "tn_t0.5": cm["tn"],
        "recall_at_1pct_fpr": rec_1,
        "threshold_at_1pct_fpr": thr_1,
        "recall_at_5pct_fpr": rec_5,
        "threshold_at_5pct_fpr": thr_5,
        "best_f1": best_f1,
        "best_f1_threshold": best_f1_t,
        "positive_rate": float(labels.mean()),
        "n_samples": int(labels.size),
    }


def save_run_artifacts(
    out_dir: Path,
    probs: np.ndarray,
    labels: np.ndarray,
    files: list[str] | None = None,
) -> dict[str, float]:
    """Save predictions.npz + PR/ROC/confusion PNGs + summary.json under ``out_dir``.

    Returns the summary dict so the caller can also log scalars to W&B.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_predictions(out_dir / "predictions.npz", probs, labels, files)

    plot_pr_curve(probs, labels).savefig(out_dir / "pr_curve.png", dpi=150)
    plot_roc_curve(probs, labels).savefig(out_dir / "roc_curve.png", dpi=150)
    plot_confusion_matrix(probs, labels, threshold=0.5).savefig(
        out_dir / "confusion_matrix_t0.5.png", dpi=150
    )

    summary = compute_summary(probs, labels)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
