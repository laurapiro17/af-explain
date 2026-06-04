"""Training CLI: ``af-train --data-dir data/raw/training2017 --epochs 50``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import lightning as L  # noqa: N812  (Lightning convention)
import torch
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from af_explain.data.dataset import PhysioNet2017Dataset
from af_explain.training.lightning_module import AFClassifier

app = typer.Typer(add_completion=False, help="Train the AFib classifier on PhysioNet 2017.")


@app.command()
def train(
    data_dir: Annotated[Path, typer.Option(help="Path to extracted training2017/")] = Path(
        "data/raw/training2017"
    ),
    output_dir: Annotated[Path, typer.Option(help="Where to save checkpoints + logs")] = Path(
        "outputs"
    ),
    epochs: Annotated[int, typer.Option(help="Maximum training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 32,
    num_workers: Annotated[int, typer.Option(help="DataLoader workers")] = 4,
    learning_rate: Annotated[float, typer.Option(help="AdamW learning rate")] = 1e-3,
    weight_decay: Annotated[float, typer.Option(help="AdamW weight decay")] = 1e-4,
    seed: Annotated[int, typer.Option(help="Reproducibility seed")] = 42,
    fast_dev_run: Annotated[bool, typer.Option(help="Run 1 batch for smoke testing")] = False,
) -> None:
    """Train + validate the AFib classifier."""
    L.seed_everything(seed, workers=True)

    train_ds = PhysioNet2017Dataset(data_dir, split="train", sample_mode="random")
    val_ds = PhysioNet2017Dataset(data_dir, split="val", sample_mode="center")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    model = AFClassifier(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weights=train_ds.class_weights,
        scheduler_t_max=epochs,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="af-{epoch:02d}-{val/f1_macro:.4f}",
            monitor="val/f1_macro",
            mode="max",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(monitor="val/f1_macro", mode="max", patience=10),
        RichProgressBar(),
    ]

    trainer = L.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        logger=CSVLogger(output_dir / "logs", name="af-explain"),
        deterministic=True,
        fast_dev_run=fast_dev_run,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    app()
