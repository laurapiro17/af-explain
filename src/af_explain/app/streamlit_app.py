"""Streamlit demo for af-explain.

Run locally::

    streamlit run src/af_explain/app/streamlit_app.py

Or visit the hosted demo at:
    https://huggingface.co/spaces/laurapiro17/af-explain   (after deploy)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import wfdb
from plotly.subplots import make_subplots

from af_explain.data.dataset import LABEL_NAMES
from af_explain.data.preprocess import DEFAULT_FS, preprocess_record
from af_explain.explain.gradcam import make_gradcam
from af_explain.explain.integrated_gradients import explain_with_ig
from af_explain.training.lightning_module import AFClassifier

st.set_page_config(
    page_title="af-explain · AFib detection with explanations",
    page_icon="❤️",
    layout="wide",
)


@st.cache_resource
def load_model(checkpoint_path: str) -> AFClassifier:
    model = AFClassifier.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    return model


def _read_uploaded(file) -> tuple[np.ndarray, int]:
    """Accept .npy, .csv, or PhysioNet (.hea + .mat) pairs."""
    suffix = Path(file.name).suffix.lower()
    if suffix == ".npy":
        return np.load(file).astype(np.float32).squeeze(), DEFAULT_FS
    if suffix == ".csv":
        arr = np.loadtxt(file, delimiter=",").astype(np.float32).squeeze()
        return arr, DEFAULT_FS
    raise ValueError(f"Unsupported extension {suffix}. Upload .npy or .csv with a single ECG lead.")


def _load_demo_record(record_id: str, root: Path) -> tuple[np.ndarray, int]:
    signal, meta = wfdb.rdsamp(str(root / record_id))
    return signal[:, 0].astype(np.float32), int(meta["fs"])


def _plot_explanation(
    raw: np.ndarray,
    processed: np.ndarray,
    saliency: np.ndarray,
    ig_attr: np.ndarray,
    fs: int,
) -> go.Figure:
    t = np.arange(len(processed)) / fs
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Preprocessed ECG",
            "Grad-CAM saliency (which time windows mattered)",
            "Integrated Gradients (per-sample attribution, signed)",
        ),
        vertical_spacing=0.07,
    )
    fig.add_trace(go.Scatter(x=t, y=processed, name="ECG", line={"color": "black"}), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=t, y=saliency, name="Grad-CAM", line={"color": "crimson"}),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=ig_attr, name="IG", line={"color": "steelblue"}),
        row=3,
        col=1,
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_layout(
        height=720,
        showlegend=False,
        margin={"t": 60, "b": 40, "l": 40, "r": 20},
    )
    return fig


def main() -> None:
    st.title("af-explain")
    st.markdown(
        "**Explainable atrial fibrillation detection on single-lead ECG.** "
        "Upload an ECG (.npy / .csv, single lead, 300 Hz) or pick a demo record."
    )

    with st.sidebar:
        st.header("Model")
        checkpoint = st.text_input(
            "Checkpoint path",
            value="outputs/checkpoints/best.ckpt",
            help="Path to a trained PyTorch Lightning checkpoint.",
        )
        st.divider()
        st.header("Explanation")
        target_class_name = st.selectbox(
            "Target class for attribution",
            options=["(predicted)", *LABEL_NAMES],
            help="What class do you want the explanation to be *about*?",
        )

    upload_tab, demo_tab = st.tabs(["Upload ECG", "Demo record"])

    raw_signal: np.ndarray | None = None
    fs: int = DEFAULT_FS

    with upload_tab:
        uploaded = st.file_uploader("Single-lead ECG (.npy or .csv)", type=["npy", "csv"])
        if uploaded is not None:
            raw_signal, fs = _read_uploaded(uploaded)

    with demo_tab:
        demo_root = Path(st.text_input("Demo dataset root", value="data/raw/training2017"))
        record_id = st.text_input("Record ID (e.g. A00001)", value="A00001")
        if st.button("Load demo record") and demo_root.exists():
            raw_signal, fs = _load_demo_record(record_id, demo_root)

    if raw_signal is None:
        st.info("Upload a file or load a demo record to get started.")
        return

    if not Path(checkpoint).exists():
        st.error(
            f"Checkpoint not found at {checkpoint}. "
            "Train a model first with `af-train`, or update the path in the sidebar."
        )
        return

    model = load_model(checkpoint)
    processed, _mask = preprocess_record(raw_signal, fs=fs)
    tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))

    target_idx = (
        pred_idx if target_class_name == "(predicted)" else LABEL_NAMES.index(target_class_name)
    )

    cam = make_gradcam(model.model)
    saliency = cam(tensor, target_class=target_idx)
    cam.remove_hooks()
    ig_attr = explain_with_ig(model.model, tensor, target_class=target_idx)

    cols = st.columns(len(LABEL_NAMES))
    for col, name, prob in zip(cols, LABEL_NAMES, probs, strict=True):
        col.metric(
            label=name + (" ← predicted" if name == LABEL_NAMES[pred_idx] else ""),
            value=f"{prob:.2%}",
        )

    st.plotly_chart(
        _plot_explanation(raw_signal, processed, saliency, ig_attr, DEFAULT_FS),
        use_container_width=True,
    )

    with st.expander("How to read these explanations"):
        st.markdown(
            """
            **Grad-CAM** highlights *which 1-second windows of the ECG* the model
            relied on for the chosen class. Tall red regions are the parts that
            most influenced the prediction.

            **Integrated Gradients** is sample-level and *signed*: positive
            values push the prediction *toward* the chosen class; negative
            values push *away*. Useful for inspecting whether the model attends
            to P waves, QRS morphology, or RR-interval irregularity.

            **Clinical caveat**: these explanations are post-hoc
            approximations of what the model attends to. They do **not** imply
            causation and should never replace a cardiologist's reading.
            """
        )


if __name__ == "__main__":
    main()
