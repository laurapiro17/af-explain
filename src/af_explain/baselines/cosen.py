"""Coefficient of Sample Entropy (COSEn) for AFib detection on RR intervals.

Reference
---------
Lake DE, Moorman JR. Accurate estimation of entropy in very short
physiological time series: the problem of atrial fibrillation detection
in implanted ventricular devices. *Am J Physiol Heart Circ Physiol*
2011;300(1):H319-25. https://doi.org/10.1152/ajpheart.00561.2010

The detector
------------
The intuition is straightforward: atrial fibrillation produces RR
intervals that look closer to a random sequence than to a sinus-rhythm
sequence, even over windows as short as twelve beats. Sample entropy
quantifies that randomness; the two adjustments below make the
quantification robust on short windows and comparable across patients
with different heart rates.

1. ``sample_entropy``
    Standard Richman & Moorman (2000) sample entropy with template
    length ``m`` and tolerance ``r``. We use the Lake & Moorman
    "minimum-numerator" convention: when no length-``m+1`` matches are
    observed, the count is replaced by 1 rather than 0, which keeps
    SampEn finite on twelve-beat windows where matches are otherwise
    sparse.

2. ``quadratic_sample_entropy`` (QSE)
    QSE = SampEn + ln(2r). The additive term cancels the (otherwise
    arbitrary) dependence on the tolerance ``r``, turning SampEn into a
    density estimate that can be compared across studies.

3. ``cosen`` = QSE - ln(mean RR)
    Subtracts the log mean RR so that two recordings with identical
    irregularity but different mean heart rate produce the same COSEn.
    Without this, slow-and-variable AF would be confused with
    fast-and-regular sinus rhythm.

Convention: RR intervals in **seconds** (matches NeuroKit2 and most
ECG libraries). Tolerance ``r`` defaults to 0.03 s (30 ms), which is the
value Lake & Moorman use for Holter recordings sampled at 128 Hz; it is
the smallest tolerance that still gives a non-trivial match rate on
twelve-beat windows.

Threshold for AF classification is recording- and population-dependent.
Lake & Moorman 2011 report ROC-AUC near 0.95 on the MIT-BIH AF Database
with a cutoff in the range -1.5 to -1.0; we default to ``-1.4`` and
expose it as a parameter so that callers can re-tune on their own data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class CoSEnResult:
    """Numerical components of a single COSEn computation.

    Attributes
    ----------
    cosen
        The coefficient of sample entropy itself. Higher → more irregular.
    qse
        Quadratic sample entropy (= ``sample_entropy + ln(2r)``).
    sample_entropy
        Raw SampEn before the QSE adjustment.
    mean_rr
        Mean RR interval over the window, in seconds.
    n_beats
        Number of RR intervals used (i.e. window length).
    m
        Template length used in the SampEn computation.
    r
        Tolerance (seconds) used in the SampEn computation.
    """

    cosen: float
    qse: float
    sample_entropy: float
    mean_rr: float
    n_beats: int
    m: int
    r: float


def sample_entropy(
    rr_intervals: ArrayLike,
    m: int = 1,
    r: float = 0.03,
    min_match: bool = True,
) -> float:
    """Sample entropy on an RR-interval series.

    Parameters
    ----------
    rr_intervals
        RR intervals in seconds.
    m
        Template length. ``m=1`` is the Lake & Moorman default for AF.
    r
        Tolerance in seconds. Two template values are considered a match
        when their Chebyshev distance is at most ``r``.
    min_match
        If True (default), apply the Lake & Moorman minimum-numerator
        convention: replace zero length-``m+1`` matches with 1 to avoid
        an undefined ``-ln(0)``. Set False to follow the strict
        Richman-Moorman definition (returns ``+inf`` instead).

    Returns
    -------
    float
        ``-ln(A/B)`` where ``B`` counts length-``m`` template matches and
        ``A`` counts length-``m+1`` matches. Returns ``nan`` when the
        series is too short (``< m + 2`` samples).
    """
    rr = np.asarray(rr_intervals, dtype=np.float64).ravel()
    n = rr.size
    if n < m + 2:
        return float("nan")

    # Richman & Moorman compare length-m and length-(m+1) over the SAME set of
    # starting positions (0 .. n-m-1) so that SampEn collapses to 0 on a
    # constant series. With m+1 templates that means n-m positions; we truncate
    # length-m templates to the same count.
    k = n - m

    def _count_matches(length: int) -> int:
        templates = np.lib.stride_tricks.sliding_window_view(rr, window_shape=length)[:k]
        diff = np.abs(templates[:, None, :] - templates[None, :, :]).max(axis=-1)
        np.fill_diagonal(diff, np.inf)
        # Each unordered pair is counted twice in the symmetric matrix.
        return int((diff <= r).sum() // 2)

    b = _count_matches(m)
    a = _count_matches(m + 1)

    if b == 0:
        return float("nan")
    if a == 0:
        if not min_match:
            return float("inf")
        a = 1

    return float(-np.log(a / b))


def quadratic_sample_entropy(
    rr_intervals: ArrayLike,
    m: int = 1,
    r: float = 0.03,
    min_match: bool = True,
) -> float:
    """Quadratic sample entropy: SampEn + ln(2r).

    The additive ``ln(2r)`` corrects for the (otherwise arbitrary)
    dependence on the tolerance choice, leaving a quantity that
    estimates density entropy comparably across studies.
    """
    se = sample_entropy(rr_intervals, m=m, r=r, min_match=min_match)
    if not np.isfinite(se):
        return se
    return se + float(np.log(2.0 * r))


def cosen(
    rr_intervals: ArrayLike,
    m: int = 1,
    r: float = 0.03,
    min_match: bool = True,
    return_components: bool = False,
) -> float | CoSEnResult:
    """Coefficient of sample entropy on an RR-interval window.

    Parameters
    ----------
    rr_intervals
        RR intervals in seconds. Twelve beats is the minimum window
        Lake & Moorman validate; shorter windows return ``nan``.
    m, r, min_match
        Forwarded to :func:`sample_entropy`.
    return_components
        If True, return a :class:`CoSEnResult` with the SampEn and QSE
        intermediates as well. Useful for debugging and for plotting
        the components separately in clinical-analysis notebooks.

    Returns
    -------
    float or CoSEnResult
        COSEn = QSE - ln(mean RR). Higher → more irregular →
        more AF-like. Returns ``nan`` (or a ``CoSEnResult`` whose
        ``cosen`` field is ``nan``) when the input is degenerate.
    """
    rr = np.asarray(rr_intervals, dtype=np.float64).ravel()
    n = rr.size
    nan_result = CoSEnResult(
        cosen=float("nan"),
        qse=float("nan"),
        sample_entropy=float("nan"),
        mean_rr=float("nan"),
        n_beats=int(n),
        m=m,
        r=r,
    )

    if n < m + 2:
        return nan_result if return_components else float("nan")

    mean_rr = float(rr.mean())
    if mean_rr <= 0:
        return nan_result if return_components else float("nan")

    se = sample_entropy(rr, m=m, r=r, min_match=min_match)
    if not np.isfinite(se):
        return nan_result if return_components else float("nan")

    qse = se + float(np.log(2.0 * r))
    value = qse - float(np.log(mean_rr))

    if return_components:
        return CoSEnResult(
            cosen=value,
            qse=qse,
            sample_entropy=se,
            mean_rr=mean_rr,
            n_beats=int(n),
            m=m,
            r=r,
        )
    return value


def detect_af_cosen(
    rr_intervals: ArrayLike,
    window_beats: int = 12,
    stride_beats: int = 1,
    threshold: float = -1.4,
    m: int = 1,
    r: float = 0.03,
    min_match: bool = True,
) -> dict[str, np.ndarray]:
    """Sliding-window AF detection.

    Parameters
    ----------
    rr_intervals
        RR intervals in seconds (entire record).
    window_beats
        Beats per analysis window. Twelve is the Lake & Moorman default.
    stride_beats
        Hop size between successive windows. ``1`` reproduces the
        original beat-by-beat reporting; larger values trade temporal
        resolution for compute.
    threshold
        COSEn cutoff above which a window is flagged as AF.
    m, r, min_match
        Forwarded to :func:`cosen`.

    Returns
    -------
    dict
        ``window_start``: index (within ``rr_intervals``) of the first
        beat of each window.
        ``cosen``: COSEn value per window (may contain NaNs).
        ``af_flag``: boolean array, True when ``cosen > threshold``.
        ``af_burden``: fraction of windows flagged as AF (scalar float,
        NaN windows excluded).
    """
    rr = np.asarray(rr_intervals, dtype=np.float64).ravel()
    n = rr.size
    if n < window_beats:
        return {
            "window_start": np.array([], dtype=np.int64),
            "cosen": np.array([], dtype=np.float64),
            "af_flag": np.array([], dtype=bool),
            "af_burden": float("nan"),
        }

    starts = np.arange(0, n - window_beats + 1, stride_beats, dtype=np.int64)
    values = np.array(
        [cosen(rr[s : s + window_beats], m=m, r=r, min_match=min_match) for s in starts],
        dtype=np.float64,
    )
    flags = values > threshold
    valid = np.isfinite(values)
    burden = float(flags[valid].mean()) if valid.any() else float("nan")

    return {
        "window_start": starts,
        "cosen": values,
        "af_flag": flags,
        "af_burden": burden,
    }
