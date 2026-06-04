"""Tests for signal preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from af_explain.data.preprocess import (
    DEFAULT_FS,
    DEFAULT_LENGTH,
    bandpass_filter,
    normalize_ecg,
    preprocess_record,
    segment_signal,
)


def test_bandpass_preserves_length(synthetic_ecg: np.ndarray) -> None:
    out = bandpass_filter(synthetic_ecg, fs=DEFAULT_FS)
    assert out.shape == synthetic_ecg.shape
    assert out.dtype == np.float32


def test_bandpass_removes_drift(synthetic_ecg: np.ndarray) -> None:
    out = bandpass_filter(synthetic_ecg, fs=DEFAULT_FS)
    # Filtered signal should have lower mean drift than raw.
    assert abs(out.mean()) <= abs(synthetic_ecg.mean()) + 1e-3


def test_normalize_zero_mean_unit_std(synthetic_ecg: np.ndarray) -> None:
    out = normalize_ecg(synthetic_ecg)
    assert abs(float(out.mean())) < 1e-4
    assert abs(float(out.std()) - 1.0) < 1e-3


@pytest.mark.parametrize("mode", ["center", "random"])
def test_segment_to_target_length(mode: str) -> None:
    rng = np.random.default_rng(0)
    short = rng.standard_normal(5000).astype(np.float32)
    long_ = rng.standard_normal(15000).astype(np.float32)
    out_short, mask_short = segment_signal(short, target_length=DEFAULT_LENGTH, mode=mode)
    out_long, mask_long = segment_signal(long_, target_length=DEFAULT_LENGTH, mode=mode)
    assert out_short.shape == (DEFAULT_LENGTH,)
    assert out_long.shape == (DEFAULT_LENGTH,)
    assert mask_short.sum() == 5000  # only real samples are unmasked
    assert mask_long.sum() == DEFAULT_LENGTH  # crops keep everything real


def test_preprocess_record_pipeline(synthetic_ecg: np.ndarray) -> None:
    signal, mask = preprocess_record(synthetic_ecg)
    assert signal.shape == (DEFAULT_LENGTH,)
    assert mask.shape == (DEFAULT_LENGTH,)
    assert signal.dtype == np.float32
