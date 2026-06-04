"""Integrated Gradients via Captum, plus a baseline-aware ECG wrapper.

Reference:
    Sundararajan M, Taly A, Yan Q. Axiomatic Attribution for Deep Networks.
    ICML 2017. https://arxiv.org/abs/1703.01365

Why two explainability methods?
    Grad-CAM is coarse but intuitive (which 1-second window mattered?).
    Integrated Gradients is sample-level, giving the contribution of every
    sample point — useful for inspecting whether the model attends to P
    waves, QRS complexes, RR-interval irregularity, or noise.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients


def explain_with_ig(
    model: nn.Module,
    signal: torch.Tensor,
    target_class: int | None = None,
    n_steps: int = 50,
    baseline: torch.Tensor | None = None,
) -> np.ndarray:
    """Compute Integrated Gradients attribution for one ECG signal.

    Args:
        model: trained model in eval mode.
        signal: tensor ``(1, T)`` or ``(B, 1, T)``. Only the first sample is used.
        target_class: class index to attribute (default: predicted class).
        n_steps: Riemann-sum steps along the path from baseline to input.
        baseline: reference signal (default: zeros, ≈ isoelectric ECG line).

    Returns:
        Per-sample attribution as a 1-D ``np.ndarray`` of length ``T``,
        sign-preserved and normalized to ``[-1, 1]``.
    """
    model.eval()
    if signal.dim() == 2:
        signal = signal.unsqueeze(0)
    if baseline is None:
        baseline = torch.zeros_like(signal)

    with torch.no_grad():
        logits = model(signal)
    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        signal,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
    )
    attr_np = attributions[0, 0].cpu().numpy()
    abs_max = np.abs(attr_np).max()
    if abs_max > 0:
        attr_np = attr_np / abs_max
    return attr_np
