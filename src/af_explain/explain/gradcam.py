"""1-D Grad-CAM for ECG models.

Reference:
    Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D.
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
    Localization. ICCV 2017. https://doi.org/10.1109/ICCV.2017.74

Adapted to 1-D ECG signals: instead of a 2-D class-activation map over an
image, we produce a 1-D saliency vector over time. The output is upsampled
to the original signal length and overlaid on the ECG so a clinician can
see which beats / segments drove the model's prediction.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM1D:
    """Compute 1-D Grad-CAM saliency for a target class.

    Usage::

        cam = GradCAM1D(model, target_layer=model.stages[-1])
        saliency = cam(signal_tensor, target_class=1)  # AFib

    The hook captures activations and gradients on the last conv stage; on
    each call we backprop the chosen logit, compute channel-wise importance
    weights (global-average-pooled gradients), and produce a ReLU-rectified
    weighted sum of activations — finally upsampled to the input length.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd_hook(_module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
            self._activations = out.detach()

        def bwd_hook(_module: nn.Module, _grad_in: tuple, grad_out: tuple) -> None:
            self._gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(
        self,
        signal: torch.Tensor,
        target_class: int | None = None,
        upsample_to: int | None = None,
    ) -> np.ndarray:
        """Return a normalized 1-D saliency vector for ``signal``.

        Args:
            signal: tensor of shape ``(B, 1, T)`` or ``(1, T)``.
            target_class: class index to explain (default: model prediction).
            upsample_to: output length (default: input length T).
        """
        self.model.eval()
        if signal.dim() == 2:
            signal = signal.unsqueeze(0)
        signal = signal.clone().requires_grad_(True)

        logits = self.model(signal)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)

        assert self._activations is not None and self._gradients is not None
        # weights = GAP over time of gradients → (B, C)
        weights = self._gradients.mean(dim=2)
        # weighted sum over channels → (B, T_feat)
        cam = (weights.unsqueeze(-1) * self._activations).sum(dim=1)
        cam = F.relu(cam)

        target_length = upsample_to or signal.shape[-1]
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

        cam_np = cam[0].cpu().numpy()
        cam_max = cam_np.max()
        if cam_max > 0:
            cam_np = cam_np / cam_max
        return cam_np


def _last_conv_layer(model: nn.Module) -> nn.Module:
    """Heuristic: pick the last ``nn.Conv1d`` in the module tree."""
    last: nn.Module | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            last = module
    if last is None:
        raise ValueError("Model contains no Conv1d layer; cannot run Grad-CAM.")
    return last


def make_gradcam(model: nn.Module, target_layer: nn.Module | Callable | None = None) -> GradCAM1D:
    """Convenience factory: auto-pick the last conv stage if not given."""
    if target_layer is None:
        target_layer = _last_conv_layer(model)
    if callable(target_layer) and not isinstance(target_layer, nn.Module):
        target_layer = target_layer(model)
    return GradCAM1D(model, target_layer=target_layer)
