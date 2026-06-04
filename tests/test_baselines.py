"""Tests for the classical baselines module."""

from __future__ import annotations

import numpy as np
import pytest

from af_explain.baselines import (
    CoSEnResult,
    cosen,
    detect_af_cosen,
    quadratic_sample_entropy,
    sample_entropy,
)

# ─── helpers ───────────────────────────────────────────────────────────────


def _sinus_rr(n: int, mean: float = 0.8, sd: float = 0.02, seed: int = 0) -> np.ndarray:
    """Regular sinus-like RR series: small Gaussian jitter around a mean."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=sd, size=n).astype(np.float64)


def _af_rr(n: int, low: float = 0.4, high: float = 1.2, seed: int = 0) -> np.ndarray:
    """AF-like RR series: uniform draws between low and high, no autocorrelation."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=n).astype(np.float64)


# ─── sample entropy ────────────────────────────────────────────────────────


def test_sample_entropy_constant_returns_low_or_nan() -> None:
    # All-equal series: at m=1, every pair matches, so A/B → 1 and SampEn → 0.
    rr = np.full(50, 0.8)
    se = sample_entropy(rr, m=1, r=0.03)
    assert np.isfinite(se)
    assert se == pytest.approx(0.0, abs=1e-9)


def test_sample_entropy_random_higher_than_regular() -> None:
    sinus = _sinus_rr(120)
    af = _af_rr(120)
    se_sinus = sample_entropy(sinus, m=1, r=0.03)
    se_af = sample_entropy(af, m=1, r=0.03)
    assert np.isfinite(se_sinus) and np.isfinite(se_af)
    assert se_af > se_sinus


def test_sample_entropy_too_short_returns_nan() -> None:
    assert np.isnan(sample_entropy(np.array([0.8, 0.82]), m=1))


def test_sample_entropy_min_match_finite_when_strict_is_inf() -> None:
    # Three 0.4 values guarantee length-1 matches (B > 0), but every length-2
    # template (0.4,*) has a unique successor, so A = 0 and strict SampEn → ∞.
    rr = np.array([0.4, 0.5, 0.4, 0.6, 0.4])
    strict = sample_entropy(rr, m=1, r=0.001, min_match=False)
    relaxed = sample_entropy(rr, m=1, r=0.001, min_match=True)
    assert np.isinf(strict)
    assert np.isfinite(relaxed)


# ─── QSE ───────────────────────────────────────────────────────────────────


def test_qse_equals_sampen_plus_log_2r() -> None:
    rr = _sinus_rr(80)
    se = sample_entropy(rr, m=1, r=0.03)
    qse = quadratic_sample_entropy(rr, m=1, r=0.03)
    assert qse == pytest.approx(se + np.log(2 * 0.03), rel=1e-12)


# ─── COSEn ─────────────────────────────────────────────────────────────────


def test_cosen_af_higher_than_sinus_short_window() -> None:
    # Twelve-beat window is the Lake & Moorman validation length.
    sinus = _sinus_rr(12, seed=1)
    af = _af_rr(12, seed=1)
    c_sinus = cosen(sinus)
    c_af = cosen(af)
    assert np.isfinite(c_sinus) and np.isfinite(c_af)
    assert c_af > c_sinus


def test_cosen_separates_distributions_over_many_windows() -> None:
    # Averaging over independent draws should make AF > sinus with margin.
    n_reps = 30
    sinus_vals = np.array([cosen(_sinus_rr(12, seed=s)) for s in range(n_reps)])
    af_vals = np.array([cosen(_af_rr(12, seed=s)) for s in range(n_reps)])
    assert np.nanmean(af_vals) > np.nanmean(sinus_vals) + 0.5


def test_cosen_returns_components_when_requested() -> None:
    rr = _af_rr(20)
    result = cosen(rr, return_components=True)
    assert isinstance(result, CoSEnResult)
    assert result.n_beats == 20
    assert result.m == 1
    assert result.r == 0.03
    assert result.mean_rr == pytest.approx(rr.mean(), rel=1e-12)
    assert result.qse == pytest.approx(result.sample_entropy + np.log(2 * 0.03), rel=1e-12)
    assert result.cosen == pytest.approx(result.qse - np.log(result.mean_rr), rel=1e-12)


def test_cosen_too_short_returns_nan() -> None:
    assert np.isnan(cosen(np.array([0.8, 0.82])))
    nan_result = cosen(np.array([0.8, 0.82]), return_components=True)
    assert isinstance(nan_result, CoSEnResult)
    assert np.isnan(nan_result.cosen)


# ─── detect_af_cosen ───────────────────────────────────────────────────────


def test_detect_af_cosen_shapes() -> None:
    rr = _af_rr(60)
    out = detect_af_cosen(rr, window_beats=12, stride_beats=1)
    expected_windows = 60 - 12 + 1
    assert out["window_start"].shape == (expected_windows,)
    assert out["cosen"].shape == (expected_windows,)
    assert out["af_flag"].shape == (expected_windows,)
    assert out["af_flag"].dtype == bool


def test_detect_af_cosen_burden_higher_for_af() -> None:
    sinus = _sinus_rr(120, seed=2)
    af = _af_rr(120, seed=2)
    out_sinus = detect_af_cosen(sinus, threshold=-1.4)
    out_af = detect_af_cosen(af, threshold=-1.4)
    assert out_af["af_burden"] > out_sinus["af_burden"]


def test_detect_af_cosen_too_few_beats_empty() -> None:
    out = detect_af_cosen(np.array([0.8, 0.9, 0.85]), window_beats=12)
    assert out["window_start"].size == 0
    assert out["cosen"].size == 0
    assert out["af_flag"].size == 0
    assert np.isnan(out["af_burden"])


def test_detect_af_cosen_stride_skips_windows() -> None:
    rr = _af_rr(60)
    out1 = detect_af_cosen(rr, window_beats=12, stride_beats=1)
    out6 = detect_af_cosen(rr, window_beats=12, stride_beats=6)
    assert out6["window_start"].size < out1["window_start"].size
    # Strided starts are exactly the every-6th subset.
    assert np.array_equal(out6["window_start"], out1["window_start"][::6])
