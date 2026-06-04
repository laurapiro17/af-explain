"""PyTorch ``Dataset`` for the PhysioNet/CinC Challenge 2017 training set."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

from af_explain.data.preprocess import DEFAULT_FS, DEFAULT_LENGTH, preprocess_record

LABEL_MAP: dict[str, int] = {
    "N": 0,  # Normal sinus rhythm
    "A": 1,  # Atrial fibrillation
    "O": 2,  # Other arrhythmia
    "~": 3,  # Noisy / unclassifiable
}
LABEL_NAMES: list[str] = ["Normal", "AFib", "Other", "Noisy"]
NUM_CLASSES: int = len(LABEL_MAP)


class PhysioNet2017Dataset(Dataset):
    """Single-lead ECG dataset from the PhysioNet/CinC Challenge 2017.

    Each example is a tuple ``(signal, mask, label)`` where ``signal`` is a
    ``(1, target_length)`` float32 tensor (channel-first, ready for ``Conv1d``),
    ``mask`` flags real vs padded samples, and ``label`` is a class index.

    Args:
        root: Directory containing the extracted ``training2017/`` folder
            (use ``download_physionet2017`` to populate it).
        split: ``"train"``, ``"val"``, or ``"test"`` — uses a deterministic
            70/15/15 stratified split seeded by ``split_seed``.
        target_length: Fixed signal length in samples (default 9000 = 30 s @ 300 Hz).
        sample_mode: ``"center"`` for deterministic crop (val/test) or
            ``"random"`` for stochastic crop (train).
        split_seed: RNG seed for the train/val/test split.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        target_length: int = DEFAULT_LENGTH,
        sample_mode: Literal["center", "random"] = "center",
        split_seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.target_length = target_length
        self.sample_mode = sample_mode

        ref_csv = self.root / "REFERENCE.csv"
        if not ref_csv.exists():
            raise FileNotFoundError(
                f"REFERENCE.csv not found in {self.root}. "
                "Did you run `python -m af_explain.data.download`?"
            )

        df = pd.read_csv(ref_csv, header=None, names=["record", "label"])
        df = df[df["label"].isin(LABEL_MAP)].reset_index(drop=True)
        df["y"] = df["label"].map(LABEL_MAP)

        self.records = self._split_dataframe(df, split=split, seed=split_seed)

    @staticmethod
    def _split_dataframe(
        df: pd.DataFrame,
        split: str,
        seed: int,
        ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        chunks: list[pd.DataFrame] = []
        for _label, group in df.groupby("y"):
            shuffled = group.sample(frac=1.0, random_state=rng.integers(1 << 31))
            n = len(shuffled)
            n_train = int(n * ratios[0])
            n_val = int(n * ratios[1])
            if split == "train":
                chunks.append(shuffled.iloc[:n_train])
            elif split == "val":
                chunks.append(shuffled.iloc[n_train : n_train + n_val])
            elif split == "test":
                chunks.append(shuffled.iloc[n_train + n_val :])
            else:
                raise ValueError(f"Unknown split: {split}")
        return pd.concat(chunks).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.records)

    def _load_signal(self, record_id: str) -> np.ndarray:
        record_path = self.root / record_id
        signal, _meta = wfdb.rdsamp(str(record_path))
        return signal[:, 0].astype(np.float32)  # single-lead

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.records.iloc[idx]
        raw = self._load_signal(row["record"])
        signal, mask = preprocess_record(
            raw,
            fs=DEFAULT_FS,
            target_length=self.target_length,
            mode=self.sample_mode,
        )
        return {
            "signal": torch.from_numpy(signal).unsqueeze(0),  # (1, T)
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "label": torch.tensor(int(row["y"]), dtype=torch.long),
            "record_id": row["record"],
        }

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for the CrossEntropyLoss (handles imbalance)."""
        counts = self.records["y"].value_counts().sort_index().to_numpy()
        weights = counts.sum() / (NUM_CLASSES * counts)
        return torch.tensor(weights, dtype=torch.float32)
