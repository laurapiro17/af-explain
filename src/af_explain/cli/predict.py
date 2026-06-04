"""``af-predict``: run inference + explanations on a single ECG file."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer

from af_explain.data.dataset import LABEL_NAMES
from af_explain.data.preprocess import DEFAULT_FS, preprocess_record
from af_explain.explain.gradcam import make_gradcam
from af_explain.explain.integrated_gradients import explain_with_ig
from af_explain.training.lightning_module import AFClassifier

app = typer.Typer(add_completion=False, help="Predict AFib + produce explanations.")


@app.command()
def predict(
    checkpoint: Annotated[Path, typer.Option(help="Path to a trained .ckpt")],
    ecg_path: Annotated[Path, typer.Option(help="ECG as .npy (single lead, 1-D)")],
    output_dir: Annotated[Path, typer.Option(help="Where to save outputs")] = Path("outputs/predict"),
    target_class: Annotated[int | None, typer.Option(help="Class to explain (default: predicted)")] = None,
) -> None:
    """Predict and dump probabilities + Grad-CAM + IG arrays to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.load(ecg_path).astype(np.float32).squeeze()
    processed, _ = preprocess_record(raw, fs=DEFAULT_FS)
    tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0)

    model = AFClassifier.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    target = target_class if target_class is not None else pred

    cam = make_gradcam(model.model)
    saliency = cam(tensor, target_class=target)
    cam.remove_hooks()
    ig = explain_with_ig(model.model, tensor, target_class=target)

    np.savez(
        output_dir / f"{ecg_path.stem}_explanation.npz",
        signal=processed,
        probs=probs,
        prediction=pred,
        target_class=target,
        gradcam=saliency,
        integrated_gradients=ig,
    )
    typer.echo(f"Prediction: {LABEL_NAMES[pred]}  probs={dict(zip(LABEL_NAMES, probs, strict=True))}")
    typer.echo(f"Saved explanation arrays to {output_dir / (ecg_path.stem + '_explanation.npz')}")


if __name__ == "__main__":
    app()
