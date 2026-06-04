"""Tests for the clinical analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from af_explain.evaluation.clinical_analysis import (
    calibration_report,
    failure_mode_taxonomy,
    subgroup_robustness_report,
)


def test_subgroup_robustness_smoke() -> None:
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 4, size=n)
    predictions = labels.copy()
    flip_idx = rng.choice(n, size=20, replace=False)
    predictions[flip_idx] = (predictions[flip_idx] + 1) % 4
    metadata = pd.DataFrame(
        {
            "signal_quality_bin": rng.choice(["clean", "noisy"], size=n),
        }
    )
    report = subgroup_robustness_report(
        predictions, labels, metadata, grouping_variable="signal_quality_bin"
    )
    assert {"group", "n", "sensitivity", "specificity"}.issubset(report.rows.columns)
    assert len(report.rows) == 2


def test_calibration_report_smoke() -> None:
    torch.manual_seed(0)
    logits = torch.randn(64, 4)
    labels = torch.randint(0, 4, (64,))
    report = calibration_report(logits, labels, n_bins=5)
    assert set(report.per_class_ece.keys()) == {"Normal", "AFib", "Other", "Noisy"}
    assert report.temperature is not None
    assert report.temperature > 0


def test_failure_taxonomy_buckets() -> None:
    labels = np.array([1, 1, 0, 3])  # AFib, AFib, Normal, Noisy
    preds = np.array([0, 1, 1, 0])  # FN, TP, FP, noise→rhythm
    record_ids = ["A1", "A2", "A3", "A4"]
    tax = failure_mode_taxonomy(preds, labels, record_ids)
    modes = set(tax.rows["failure_mode"])
    assert "false_negative_afib" in modes
    assert "false_positive_afib" in modes
    assert "noise_to_rhythm" in modes
