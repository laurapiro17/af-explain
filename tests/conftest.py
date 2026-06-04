"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from af_explain.data.preprocess import DEFAULT_LENGTH


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_ecg(rng: np.random.Generator) -> np.ndarray:
    """Synthetic single-lead ECG: sum of low-freq drift + sinusoidal P-QRS-T-ish."""
    fs = 300
    duration = 30
    t = np.arange(fs * duration) / fs
    drift = 0.3 * np.sin(2 * np.pi * 0.2 * t)
    qrs = 2.0 * np.sin(2 * np.pi * 1.2 * t) * np.exp(-(((t % (1 / 1.2)) - 0.05) ** 2) / 0.001)
    noise = 0.05 * rng.standard_normal(t.shape)
    return (drift + qrs + noise).astype(np.float32)


@pytest.fixture
def tiny_batch() -> dict[str, torch.Tensor]:
    """A 2-sample minibatch shaped for the model."""
    torch.manual_seed(0)
    return {
        "signal": torch.randn(2, 1, DEFAULT_LENGTH),
        "mask": torch.ones(2, 1, DEFAULT_LENGTH),
        "label": torch.tensor([0, 1], dtype=torch.long),
    }
