"""Smoke tests for the 1-D ResNet."""

from __future__ import annotations

import torch

from af_explain.data.dataset import NUM_CLASSES
from af_explain.data.preprocess import DEFAULT_LENGTH
from af_explain.models.resnet1d import ResNet1D, resnet18_1d


def test_forward_shape() -> None:
    model = resnet18_1d(num_classes=NUM_CLASSES)
    x = torch.randn(2, 1, DEFAULT_LENGTH)
    out = model(x)
    assert out.shape == (2, NUM_CLASSES)


def test_feature_map_cached() -> None:
    model = resnet18_1d()
    x = torch.randn(1, 1, DEFAULT_LENGTH)
    _ = model(x)
    assert model.feature_map is not None
    assert model.feature_map.shape[0] == 1  # batch size


def test_backward_runs() -> None:
    model = ResNet1D(num_classes=4, base_channels=16, block_counts=(1, 1))
    x = torch.randn(2, 1, 1500)
    y = torch.tensor([0, 1])
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
