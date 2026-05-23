"""Tests for the FP/FN analysis module.

Covers the metric correctness (recall@FPR, threshold sweep, confusion matrix)
and the predictions cache round-trip. Plot tests are intentionally skipped —
they're hard to assert about meaningfully; a smoke 'does it not crash' check
is captured indirectly by ``save_run_artifacts`` below.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from narw_classifier.analysis.artifacts import compute_summary, save_run_artifacts
from narw_classifier.analysis.metrics import (
    best_f1_from_sweep,
    confusion_at_threshold,
    confusion_metrics_at_threshold,
    recall_at_fpr,
    threshold_sweep,
)
from narw_classifier.analysis.predictions import load_predictions, save_predictions


def _toy_predictions(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """80 negatives + 20 positives; probs correlate with the label (signal + noise)."""
    rng = np.random.default_rng(seed)
    labels = np.array([0] * 80 + [1] * 20, dtype=np.int64)
    probs = rng.uniform(0.0, 0.5, size=80).tolist() + rng.uniform(0.5, 1.0, size=20).tolist()
    return np.array(probs, dtype=np.float32), labels


class TestRecallAtFPR:
    def test_perfect_classifier_hits_full_recall(self):
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        recall, _ = recall_at_fpr(probs, labels, target_fpr=0.0)
        assert recall == 1.0

    def test_random_classifier_recall_near_target_fpr(self):
        rng = np.random.default_rng(0)
        labels = np.repeat([0, 1], 500)
        probs = rng.uniform(0.0, 1.0, 1000).astype(np.float32)
        recall, _ = recall_at_fpr(probs, labels, target_fpr=0.1)
        # For random predictions at FPR=0.1, expect recall around 0.1 ± noise.
        assert 0.0 < recall < 0.25

    def test_invalid_target_fpr_raises(self):
        probs, labels = _toy_predictions()
        with pytest.raises(ValueError):
            recall_at_fpr(probs, labels, target_fpr=1.5)
        with pytest.raises(ValueError):
            recall_at_fpr(probs, labels, target_fpr=-0.1)


class TestConfusionAtThreshold:
    def test_layout_is_tn_fp_fn_tp(self):
        probs = np.array([0.1, 0.6, 0.4, 0.9])
        labels = np.array([0, 0, 1, 1])
        # threshold 0.5: preds = [0, 1, 0, 1] → tn=1, fp=1, fn=1, tp=1
        cm = confusion_at_threshold(probs, labels, threshold=0.5)
        assert cm.tolist() == [[1, 1], [1, 1]]

    def test_threshold_one_classifies_everything_negative(self):
        probs = np.array([0.1, 0.6, 0.4, 0.9])
        labels = np.array([0, 0, 1, 1])
        cm = confusion_at_threshold(probs, labels, threshold=1.01)
        # all predictions 0 → tn=2, fp=0, fn=2, tp=0
        assert cm.tolist() == [[2, 0], [2, 0]]


class TestThresholdSweep:
    def test_returns_all_expected_keys(self):
        probs, labels = _toy_predictions()
        result = threshold_sweep(probs, labels)
        expected = {"thresholds", "precision", "recall", "fpr", "accuracy", "f1"}
        assert set(result.keys()) == expected

    def test_recall_decreases_monotonically_with_threshold(self):
        probs, labels = _toy_predictions()
        result = threshold_sweep(probs, labels)
        # Recall is non-increasing as threshold rises (higher bar to call positive).
        assert (np.diff(result["recall"]) <= 1e-9).all()


class TestComputeSummary:
    def test_summary_has_expected_keys(self):
        probs, labels = _toy_predictions()
        summary = compute_summary(probs, labels)
        for k in (
            # ranking metrics
            "auroc",
            "ap",
            # threshold-0.5 metrics
            "accuracy_t0.5",
            "precision_t0.5",
            "recall_t0.5",
            "f1_t0.5",
            "fpr_t0.5",
            # confusion counts at threshold 0.5
            "tp_t0.5",
            "fp_t0.5",
            "fn_t0.5",
            "tn_t0.5",
            # operating-point metrics
            "recall_at_1pct_fpr",
            "threshold_at_1pct_fpr",
            "recall_at_5pct_fpr",
            "threshold_at_5pct_fpr",
            "best_f1",
            "best_f1_threshold",
            # dataset stats
            "n_samples",
            "positive_rate",
        ):
            assert k in summary

    def test_n_samples_and_positive_rate(self):
        probs, labels = _toy_predictions()
        summary = compute_summary(probs, labels)
        assert summary["n_samples"] == 100
        assert summary["positive_rate"] == pytest.approx(0.2)

    def test_confusion_counts_sum_to_n_samples(self):
        probs, labels = _toy_predictions()
        s = compute_summary(probs, labels)
        assert s["tp_t0.5"] + s["fp_t0.5"] + s["fn_t0.5"] + s["tn_t0.5"] == s["n_samples"]


class TestConfusionMetricsAtThreshold:
    def test_perfect_classifier_metrics(self):
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        m = confusion_metrics_at_threshold(probs, labels, threshold=0.5)
        assert m["tp"] == 2 and m["tn"] == 2
        assert m["fp"] == 0 and m["fn"] == 0
        assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


class TestBestF1FromSweep:
    def test_returns_threshold_in_range(self):
        probs, labels = _toy_predictions()
        f1, t = best_f1_from_sweep(probs, labels)
        assert 0.0 <= f1 <= 1.0
        assert 0.0 <= t <= 1.0


class TestPredictionsCache:
    def test_roundtrip_with_files(self, tmp_path: Path):
        probs, labels = _toy_predictions()
        files = [f"file_{i:03d}.aif" for i in range(len(probs))]
        out = tmp_path / "predictions.npz"
        save_predictions(out, probs, labels, files)
        p_r, l_r, f_r = load_predictions(out)
        assert np.allclose(p_r, probs)
        assert np.array_equal(l_r, labels)
        assert f_r == files

    def test_roundtrip_without_files(self, tmp_path: Path):
        probs, labels = _toy_predictions()
        out = tmp_path / "predictions.npz"
        save_predictions(out, probs, labels, files=None)
        _, _, f_r = load_predictions(out)
        assert f_r is None

    def test_files_length_mismatch_raises(self, tmp_path: Path):
        probs, labels = _toy_predictions()
        with pytest.raises(ValueError, match="files length"):
            save_predictions(tmp_path / "x.npz", probs, labels, files=["a.aif"])


class TestSaveRunArtifacts:
    def test_writes_all_expected_files_and_returns_summary(self, tmp_path: Path):
        probs, labels = _toy_predictions()
        files = [f"x_{i}.aif" for i in range(len(probs))]
        summary = save_run_artifacts(tmp_path, probs, labels, files)
        for name in (
            "predictions.npz",
            "pr_curve.png",
            "roc_curve.png",
            "confusion_matrix_t0.5.png",
            "summary.json",
        ):
            assert (tmp_path / name).exists(), f"missing {name}"
        assert (
            json.loads((tmp_path / "summary.json").read_text())["n_samples"] == summary["n_samples"]
        )
