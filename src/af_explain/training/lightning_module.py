"""PyTorch Lightning module for AFib classification."""

from __future__ import annotations

from typing import Any

import lightning as L  # noqa: N812  (Lightning convention)
import torch
import torch.nn as nn
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from af_explain.data.dataset import LABEL_NAMES, NUM_CLASSES
from af_explain.models.resnet1d import resnet18_1d


class AFClassifier(L.LightningModule):
    """Lightning wrapper around a 1-D ResNet for ECG rhythm classification.

    Tracks the metrics that matter clinically:
        - macro-F1: balanced score across rhythm classes (the PhysioNet 2017
          challenge headline metric).
        - per-class AUROC: how well each rhythm is separated from the rest.
        - per-class recall (sensitivity) and precision: false-negative AFib
          is the dangerous failure mode → we report it explicitly.
        - confusion matrix: which pairs of rhythms the model confuses.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_classes: int = NUM_CLASSES,
        class_weights: torch.Tensor | None = None,
        scheduler_t_max: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.model = resnet18_1d(num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        metric_kwargs = {"num_classes": num_classes, "average": "macro"}
        self.train_f1 = MulticlassF1Score(**metric_kwargs)
        self.val_f1 = MulticlassF1Score(**metric_kwargs)
        self.test_f1 = MulticlassF1Score(**metric_kwargs)

        self.val_auroc = MulticlassAUROC(num_classes=num_classes, average=None)
        self.val_recall = MulticlassRecall(num_classes=num_classes, average=None)
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average=None)

        self.test_auroc = MulticlassAUROC(num_classes=num_classes, average=None)
        self.test_recall = MulticlassRecall(num_classes=num_classes, average=None)
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average=None)
        self.test_confmat = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self(batch["signal"])
        loss = self.loss_fn(logits, batch["label"])
        return loss, logits, batch["label"]

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, logits, y = self._shared_step(batch)
        self.train_f1.update(logits, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/f1_macro", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, logits, y = self._shared_step(batch)
        self.val_f1.update(logits, y)
        self.val_auroc.update(logits, y)
        self.val_recall.update(logits, y)
        self.val_precision.update(logits, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val/f1_macro", self.val_f1.compute(), prog_bar=True)
        self._log_per_class("val", self.val_auroc.compute(), prefix="auroc")
        self._log_per_class("val", self.val_recall.compute(), prefix="recall")
        self._log_per_class("val", self.val_precision.compute(), prefix="precision")
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_recall.reset()
        self.val_precision.reset()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, logits, y = self._shared_step(batch)
        self.test_f1.update(logits, y)
        self.test_auroc.update(logits, y)
        self.test_recall.update(logits, y)
        self.test_precision.update(logits, y)
        self.test_confmat.update(logits, y)
        self.log("test/loss", loss)

    def on_test_epoch_end(self) -> None:
        self.log("test/f1_macro", self.test_f1.compute())
        self._log_per_class("test", self.test_auroc.compute(), prefix="auroc")
        self._log_per_class("test", self.test_recall.compute(), prefix="recall")
        self._log_per_class("test", self.test_precision.compute(), prefix="precision")
        # confusion matrix logged via callback or extracted in test scripts.

    def _log_per_class(self, stage: str, values: torch.Tensor, prefix: str) -> None:
        for idx, name in enumerate(LABEL_NAMES):
            self.log(f"{stage}/{prefix}_{name}", values[idx].item())

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.scheduler_t_max
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
