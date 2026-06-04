"""Signal preprocessing for single-lead ECG.

Pipeline rationale (clinical motivation):
    1. Bandpass 0.5-40 Hz: removes baseline wander (respiration, motion) and
       high-frequency noise (EMG, mains 50/60 Hz), preserving P/QRS/T morphology.
    2. Z-score normalization per record: makes the model invariant to amplitude
       differences between leads / patients / devices.
    3. Fixed-length segmentation (default 30 s @ 300 Hz = 9000 samples): the
       PhysioNet 2017 protocol — short recordings are pad-and-mask, long ones
       are center-cropped (we leave random cropping to the augmentation step).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

DEFAULT_FS = 300  # PhysioNet/CinC 2017 sampling frequency (Hz)
DEFAULT_DURATION_S = 30
DEFAULT_LENGTH = DEFAULT_FS * DEFAULT_DURATION_S  # 9000 samples


def bandpass_filter(
    signal: np.ndarray,
    fs: int = DEFAULT_FS,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Uses ``sosfiltfilt`` to avoid phase distortion — important for ECG because
    the QRS-T relative timing carries diagnostic information.
    """
    nyq = fs / 2
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def normalize_ecg(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-record z-score normalization (mean 0, std 1)."""
    mean = signal.mean()
    std = signal.std() + eps
    return ((signal - mean) / std).astype(np.float32)


def segment_signal(
    signal: np.ndarray,
    target_length: int = DEFAULT_LENGTH,
    mode: str = "center",
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a 1-D signal to ``target_length`` by cropping or zero-padding.

    Args:
        signal: 1-D array of arbitrary length.
        target_length: desired length in samples.
        mode: "center" (deterministic crop / symmetric pad) or "random"
            (random crop / pad, useful for training).

    Returns:
        (resized_signal, valid_mask) where ``valid_mask`` is 1 on real samples
        and 0 on padding — the model can use it to ignore padded regions.
    """
    n = signal.shape[0]
    mask = np.ones(target_length, dtype=np.float32)

    if n == target_length:
        return signal.astype(np.float32), mask

    if n > target_length:
        if mode == "random":
            start = np.random.randint(0, n - target_length + 1)
        else:
            start = (n - target_length) // 2
        return signal[start : start + target_length].astype(np.float32), mask

    # n < target_length → pad
    pad_total = target_length - n
    pad_left = np.random.randint(0, pad_total + 1) if mode == "random" else pad_total // 2
    pad_right = pad_total - pad_left
    padded = np.pad(signal, (pad_left, pad_right), mode="constant").astype(np.float32)
    mask[:pad_left] = 0.0
    mask[pad_left + n :] = 0.0
    return padded, mask


def preprocess_record(
    raw_signal: np.ndarray,
    fs: int = DEFAULT_FS,
    target_length: int = DEFAULT_LENGTH,
    mode: str = "center",
) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing pipeline: bandpass → normalize → segment."""
    filtered = bandpass_filter(raw_signal.astype(np.float32), fs=fs)
    normalized = normalize_ecg(filtered)
    return segment_signal(normalized, target_length=target_length, mode=mode)
