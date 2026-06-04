"""Classical (pre-deep-learning) baselines for AFib detection.

Used as reference points to quantify the added value of the 1-D ResNet:
deep models that fail to beat a 30-line entropy detector are not worth
shipping. Each baseline operates on RR-interval series rather than on raw
ECG, so peak detection is the caller's responsibility (NeuroKit2,
``wfdb``, or any other source of R-peak indices).

Currently included:
    - ``cosen``: Coefficient of Sample Entropy (Lake & Moorman 2011)
"""

from __future__ import annotations

from af_explain.baselines.cosen import (
    CoSEnResult,
    cosen,
    detect_af_cosen,
    quadratic_sample_entropy,
    sample_entropy,
)

__all__ = [
    "CoSEnResult",
    "cosen",
    "detect_af_cosen",
    "quadratic_sample_entropy",
    "sample_entropy",
]
