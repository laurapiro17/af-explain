"""Clinical analyses that turn an ECG classifier into a research artefact.

This module is the *clinical contribution* of af-explain. Anyone can train a
1-D ResNet on PhysioNet 2017. The contribution here is the four analyses
below, written from the perspective of a medical student who works on
atrial fibrillation clinically and computationally.

The four analyses
-----------------
1. ``subgroup_robustness_report``
    Audits false-negative AFib rates across stratifying variables (e.g.
    signal quality bin, recording length). Goal: surface systematic
    failure modes that an aggregate macro-F1 would hide.

2. ``calibration_report``
    Reliability diagram + expected calibration error (ECE) + optional
    temperature scaling. A clinically deployable classifier needs
    calibrated probabilities, not just high accuracy.

3. ``failure_mode_taxonomy``
    Classifies misclassifications into clinically meaningful buckets
    (low amplitude AF, atrial flutter mimic, ectopy storm, baseline
    wander, etc.) so that the discussion section can argue *why* the
    model fails rather than just *how often*.

4. ``clinician_concordance``  (semi-automatic — needs manual annotations)
    Compares Grad-CAM saliency peaks against expert-annotated regions
    of interest (P waves, RR irregularity windows). Annotations are
    stored as JSON; see ``annotations/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812  (PyTorch convention)
from sklearn.calibration import calibration_curve

from af_explain.data.dataset import LABEL_NAMES

# ─────────────────────────────────────────────────────────────────────────────
#  THE RESEARCH QUESTION
#  ↓↓↓ THIS IS WHAT LAURA WRITES (5-10 lines) ↓↓↓
#  This string is loaded into the README and the Quarto methods doc.
#  It frames the entire repository — be specific, be honest about scope.
# ─────────────────────────────────────────────────────────────────────────────
RESEARCH_QUESTION: str = """\
Background:
    Atrial fibrillation is the most common sustained arrhythmia and a
    leading cause of cardioembolic stroke. Deep-learning classifiers now
    reach cardiologist-level accuracy on single-lead ECG, yet three gaps
    block clinical and regulatory adoption: (i) aggregate metrics mask
    systematic subgroup failures, (ii) post-hoc saliency maps are rarely
    audited against the morphological features clinicians actually use,
    and (iii) misclassifications are reported as numbers rather than as
    clinically meaningful failure modes.

Question:
    For a 1-D ResNet trained on the PhysioNet/CinC 2017 single-lead ECG
    corpus, does the model
        (1) maintain its atrial-fibrillation sensitivity across
            stratifying variables such as signal quality and record
            length,
        (2) produce Grad-CAM and Integrated-Gradients saliency that
            overlaps with clinician-annotated regions of interest
            (absent P waves, irregular RR intervals, fibrillatory
            baseline), and
        (3) commit errors that can be organised into a clinically
            grounded taxonomy of failure modes (false-negative AFib,
            atrial flutter mistaken for AFib, noise read as rhythm,
            low-amplitude AF read as sinus) rather than appearing
            as undifferentiated noise in a confusion matrix?

Why this matters:
    Subgroup audit is a prerequisite for FDA/EMA submission of AI-based
    ECG tools; clinician-model concordance is a prerequisite for
    bedside trust; a clinical failure-mode taxonomy is what tells the
    next researcher where to invest data-collection and model-improvement
    effort. Together they turn an accurate classifier into a deployable
    one — and the three analyses are far cheaper to produce jointly than
    sequentially.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  1. SUBGROUP ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────────

AFIB_CLASS_INDEX: int = 1  # see LABEL_NAMES


@dataclass
class SubgroupReport:
    """Per-subgroup AFib sensitivity (recall) report."""

    grouping_variable: str
    rows: pd.DataFrame  # columns: group, n, n_afib, sensitivity, specificity, support

    def worst_group(self) -> pd.Series:
        return self.rows.sort_values("sensitivity").iloc[0]

    def to_markdown(self) -> str:
        return self.rows.to_markdown(index=False, floatfmt=".3f")


def subgroup_robustness_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    grouping_variable: str,
) -> SubgroupReport:
    """Compute AFib sensitivity/specificity stratified by a metadata column.

    Args:
        predictions: integer class predictions, shape ``(N,)``.
        labels: integer class labels, shape ``(N,)``.
        metadata: dataframe with one row per record; must contain
            ``grouping_variable`` (e.g. ``"signal_quality_bin"``) and align
            row-wise with ``predictions`` / ``labels``.
        grouping_variable: column name to stratify by.
    """
    if grouping_variable not in metadata.columns:
        raise KeyError(f"Metadata has no column {grouping_variable!r}")

    df = metadata.copy()
    df["pred"] = predictions
    df["label"] = labels
    df["is_afib"] = df["label"] == AFIB_CLASS_INDEX
    df["pred_afib"] = df["pred"] == AFIB_CLASS_INDEX

    rows = []
    for group, sub in df.groupby(grouping_variable):
        tp = int(((sub["pred_afib"]) & (sub["is_afib"])).sum())
        fn = int(((~sub["pred_afib"]) & (sub["is_afib"])).sum())
        tn = int(((~sub["pred_afib"]) & (~sub["is_afib"])).sum())
        fp = int(((sub["pred_afib"]) & (~sub["is_afib"])).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        rows.append(
            {
                "group": group,
                "n": len(sub),
                "n_afib": int(sub["is_afib"].sum()),
                "sensitivity": sens,
                "specificity": spec,
                "support": tp + fn,
            }
        )
    return SubgroupReport(grouping_variable=grouping_variable, rows=pd.DataFrame(rows))


# ─────────────────────────────────────────────────────────────────────────────
#  2. CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CalibrationReport:
    """Calibration diagnostics for a multiclass model."""

    per_class_ece: dict[str, float]
    reliability_curves: dict[str, tuple[np.ndarray, np.ndarray]]  # (prob_pred, prob_true)
    temperature: float | None = None

    def to_markdown(self) -> str:
        rows = [{"class": name, "ECE": ece} for name, ece in self.per_class_ece.items()]
        return pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f")


def expected_calibration_error(probs: np.ndarray, hits: np.ndarray, n_bins: int = 15) -> float:
    """Multiclass-friendly expected calibration error for one class."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for low, high in pairwise(bin_edges):
        mask = (probs > low) & (probs <= high)
        if not mask.any():
            continue
        bin_conf = probs[mask].mean()
        bin_acc = hits[mask].mean()
        ece += (mask.mean()) * abs(bin_conf - bin_acc)
    return float(ece)


def calibration_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> CalibrationReport:
    """One-vs-rest reliability + ECE + temperature suggestion."""
    probs = F.softmax(logits, dim=1).cpu().numpy()
    y = labels.cpu().numpy()
    per_class_ece: dict[str, float] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cls_idx, name in enumerate(LABEL_NAMES):
        cls_probs = probs[:, cls_idx]
        cls_hits = (y == cls_idx).astype(int)
        per_class_ece[name] = expected_calibration_error(cls_probs, cls_hits, n_bins=n_bins)
        prob_true, prob_pred = calibration_curve(cls_hits, cls_probs, n_bins=n_bins)
        curves[name] = (prob_pred, prob_true)

    temperature = _fit_temperature(logits, labels)
    return CalibrationReport(
        per_class_ece=per_class_ece,
        reliability_curves=curves,
        temperature=temperature,
    )


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Single-parameter temperature scaling (Guo et al. 2017)."""
    temperature = torch.nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nll(logits / temperature.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(temperature.detach().clamp(min=1e-3).item())


# ─────────────────────────────────────────────────────────────────────────────
#  3. FAILURE MODE TAXONOMY (heuristic, extensible)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FailureMode:
    """A clinically meaningful misclassification bucket."""

    name: str
    description: str
    rule: str  # human-readable condition; the actual predicate is in CODE


@dataclass
class FailureTaxonomy:
    """Categorize misclassified records into clinical buckets."""

    rows: pd.DataFrame  # one row per misclassified record
    modes: list[FailureMode] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        return (
            self.rows.groupby("failure_mode")
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )


DEFAULT_FAILURE_MODES: list[FailureMode] = [
    FailureMode(
        name="false_negative_afib",
        description="True AFib classified as Normal or Other — the dangerous error.",
        rule="label == AF and pred in {Normal, Other}",
    ),
    FailureMode(
        name="false_positive_afib",
        description="Non-AFib classified as AFib — leads to unnecessary anticoagulation work-up.",
        rule="label != AF and pred == AF",
    ),
    FailureMode(
        name="noise_to_rhythm",
        description="Noisy signal classified as a clinical rhythm — over-interpretation of artifact.",
        rule="label == Noisy and pred in {Normal, AF, Other}",
    ),
    FailureMode(
        name="other_arrhythmia_confused_with_afib",
        description="Atrial flutter / ectopy storms mistaken for AFib (mimics with irregular RR).",
        rule="label == Other and pred == AF",
    ),
]


def failure_mode_taxonomy(
    predictions: np.ndarray,
    labels: np.ndarray,
    record_ids: list[str],
    modes: list[FailureMode] | None = None,
) -> FailureTaxonomy:
    """Bucket each misclassified record into one of the failure modes."""
    modes = modes or DEFAULT_FAILURE_MODES
    rows = []
    for rec, y, yhat in zip(record_ids, labels, predictions, strict=True):
        if y == yhat:
            continue
        rows.append(
            {
                "record_id": rec,
                "label": LABEL_NAMES[int(y)],
                "pred": LABEL_NAMES[int(yhat)],
                "failure_mode": _assign_failure_mode(int(y), int(yhat)),
            }
        )
    return FailureTaxonomy(rows=pd.DataFrame(rows), modes=modes)


def _assign_failure_mode(label: int, pred: int) -> str:
    if label == 1 and pred in {0, 2}:
        return "false_negative_afib"
    if label != 1 and pred == 1 and label != 3:
        return "false_positive_afib"
    if label == 3 and pred in {0, 1, 2}:
        return "noise_to_rhythm"
    if label == 2 and pred == 1:
        return "other_arrhythmia_confused_with_afib"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
#  4. CLINICIAN-MODEL CONCORDANCE  (needs manual annotations)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConcordanceResult:
    """How well Grad-CAM peaks overlap with expert-annotated regions of interest."""

    iou: float  # intersection-over-union between saliency mask and ROI mask
    peak_inside_roi_rate: float  # fraction of top-k saliency peaks inside any ROI
    n_records: int


def clinician_concordance(
    saliency_per_record: dict[str, np.ndarray],
    annotations_dir: Path | str,
    saliency_threshold: float = 0.5,
    top_k_peaks: int = 5,
) -> ConcordanceResult:
    """Compare model saliency with expert-annotated regions of interest.

    Annotations are JSON files in ``annotations_dir`` named ``<record_id>.json``
    with structure::

        {
          "record_id": "A00001",
          "fs": 300,
          "rois": [
            {"start_s": 1.2, "end_s": 1.5, "label": "P_wave"},
            {"start_s": 4.0, "end_s": 6.0, "label": "RR_irregularity"}
          ]
        }
    """
    import json

    annotations_dir = Path(annotations_dir)
    ious: list[float] = []
    peak_hits: list[float] = []

    for record_id, saliency in saliency_per_record.items():
        ann_path = annotations_dir / f"{record_id}.json"
        if not ann_path.exists():
            continue
        ann = json.loads(ann_path.read_text())
        fs = ann["fs"]
        roi_mask = np.zeros_like(saliency, dtype=bool)
        for roi in ann["rois"]:
            start = int(roi["start_s"] * fs)
            end = int(roi["end_s"] * fs)
            roi_mask[start:end] = True

        sal_mask = saliency >= saliency_threshold
        if roi_mask.any() or sal_mask.any():
            union = (roi_mask | sal_mask).sum()
            intersection = (roi_mask & sal_mask).sum()
            ious.append(float(intersection) / float(union) if union > 0 else 0.0)

        peak_idx = np.argpartition(-saliency, top_k_peaks)[:top_k_peaks]
        peak_hits.append(float(roi_mask[peak_idx].mean()))

    if not ious:
        return ConcordanceResult(iou=float("nan"), peak_inside_roi_rate=float("nan"), n_records=0)

    return ConcordanceResult(
        iou=float(np.mean(ious)),
        peak_inside_roi_rate=float(np.mean(peak_hits)),
        n_records=len(ious),
    )
