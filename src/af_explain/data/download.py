"""Download the PhysioNet/CinC Challenge 2017 training set.

Dataset reference:
    Clifford GD, Liu C, Moody B, Lehman L, Silva I, Li Q, Johnson AEW, Mark RG.
    AF Classification from a Short Single Lead ECG Recording: the PhysioNet/Computing
    in Cardiology Challenge 2017. Computing in Cardiology 44, 2017.
    https://physionet.org/content/challenge-2017/1.0.0/

Records: 8,528 single-lead ECG recordings, 9-60 s, 300 Hz, 16-bit.
Labels:  Normal (N), Atrial Fibrillation (A), Other rhythm (O), Noisy (~).
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

console = Console()

PHYSIONET_2017_URL = "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip?download"
PHYSIONET_2017_MD5 = "5bc5b9c8b29d4eafad8e1bd5db1e8b6f"  # verify after first download
ARCHIVE_NAME = "training2017.zip"
EXTRACTED_DIR = "training2017"


def _md5sum(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_physionet2017(
    dest_dir: Path | str = "data/raw",
    verify_checksum: bool = False,
    force: bool = False,
) -> Path:
    """Download and extract the PhysioNet/CinC 2017 training set.

    Args:
        dest_dir: Where to download. The archive will go to ``dest_dir/training2017.zip``
            and be extracted into ``dest_dir/training2017/``.
        verify_checksum: If True, verify the MD5 hash of the archive.
        force: If True, re-download even if the archive already exists.

    Returns:
        Path to the extracted directory.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    archive_path = dest / ARCHIVE_NAME
    extracted_path = dest / EXTRACTED_DIR

    if extracted_path.exists() and not force:
        console.log(f"[green]✓[/green] Dataset already extracted at {extracted_path}")
        return extracted_path

    if not archive_path.exists() or force:
        console.log(f"Downloading PhysioNet 2017 archive to {archive_path}")
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("download", total=None)

            def _hook(block_num: int, block_size: int, total_size: int) -> None:
                if total_size > 0:
                    progress.update(task, total=total_size)
                progress.update(task, completed=block_num * block_size)

            urlretrieve(PHYSIONET_2017_URL, archive_path, _hook)

    if verify_checksum:
        observed = _md5sum(archive_path)
        if observed != PHYSIONET_2017_MD5:
            console.print(
                f"[yellow]Warning:[/yellow] MD5 mismatch (expected {PHYSIONET_2017_MD5}, "
                f"got {observed}). Update PHYSIONET_2017_MD5 after first verified download."
            )

    console.log(f"Extracting {archive_path}")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(dest)

    console.log(f"[green]✓[/green] Extracted to {extracted_path}")
    return extracted_path


if __name__ == "__main__":
    download_physionet2017()
