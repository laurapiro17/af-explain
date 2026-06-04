"""Tests for explainability methods."""

from __future__ import annotations

import numpy as np
import torch

from af_explain.data.preprocess import DEFAULT_LENGTH
from af_explain.explain.gradcam import make_gradcam
from af_explain.explain.integrated_gradients import explain_with_ig
from af_explain.models.resnet1d import ResNet1D


def _tiny_model() -> ResNet1D:
    return ResNet1D(num_classes=4, base_channels=8, block_counts=(1, 1), kernel_size=7)


def test_gradcam_output_shape_and_range() -> None:
    model = _tiny_model()
    x = torch.randn(1, 1, DEFAULT_LENGTH)
    cam = make_gradcam(model)
    saliency = cam(x, target_class=1)
    cam.remove_hooks()
    assert saliency.shape == (DEFAULT_LENGTH,)
    assert saliency.dtype == np.float32 or saliency.dtype == np.float64
    assert saliency.min() >= 0.0  # ReLU output
    assert saliency.max() <= 1.0 + 1e-6  # normalized


def test_integrated_gradients_signed() -> None:
    model = _tiny_model()
    x = torch.randn(1, 1, DEFAULT_LENGTH)
    attr = explain_with_ig(model, x, target_class=0, n_steps=8)
    assert attr.shape == (DEFAULT_LENGTH,)
    assert np.abs(attr).max() <= 1.0 + 1e-6  # normalized
