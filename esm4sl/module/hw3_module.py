from collections import defaultdict
from typing import Any
import os

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage
from omegaconf import DictConfig
import numpy as np
import pandas as pd

from coach_pl.configuration import configurable
from coach_pl.model import build_model, build_criterion
from coach_pl.module import MODULE_REGISTRY
from esm4sl.utils.metric import mean_loss, compute_metrics


@MODULE_REGISTRY.register()
class HW3Module(pl.LightningModule):
    @configurable
    def __init__(self, model: nn.Module, cfg: DictConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.model = torch.compile(model) if cfg.MODULE.COMPILE else model

        self.batch_size = cfg.DATALOADER.BATCH_SIZE
        self.base_lr = cfg.MODULE.OPTIMIZER.BASE_LR
        self.monitor = cfg.TRAINER.CHECKPOINT.MONITOR

        self.save_hyperparameters(ignore=['model', 'criterion'])

        self.loss_func = nn.BCEWithLogitsLoss()

        self.losses = defaultdict(list)
        self.logits = defaultdict(list)
        self.labels = defaultdict(list)
        self.indices = defaultdict(list)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> dict[str, Any]:
        model = build_model(cfg)
        return {
            "model": model,
            "cfg": cfg,
        }

    def configure_gradient_clipping(self, optimizer: Optimizer, gradient_clip_val: int | float | None = None, gradient_clip_algorithm: str | None = None) -> None:
        if gradient_clip_algorithm == "value":
            nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), gradient_clip_val)
        elif gradient_clip_algorithm == "norm":
            nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip_val)

    def configure_optimizers(self) -> dict[str, Optimizer | dict[str, str | Optimizer]]:
        lr = self.base_lr * self.batch_size * self.trainer.accumulate_grad_batches * self.trainer.world_size / 32

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0001)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.2)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            } 
        }

    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if len(batch) == 6:
            item_id, g1, g1_mask, g2, g2_mask, label = batch[0].cuda(), batch[1].cuda(), \
                batch[2].cuda(), batch[3].cuda(), batch[4].cuda(), batch[5].cuda()
        else:
            item_id, g1, g1_mask, g2, g2_mask = batch[0].cuda(), batch[1].cuda(), \
                batch[2].cuda(), batch[3].cuda(), batch[4].cuda()

        logit, attns = self.model(g1, g1_mask, g2, g2_mask)

        if len(batch) == 6:
            loss = self.loss_func(logit, label)
            return item_id, loss, logit, label
        else:
            return item_id, logit

    def step_common(self, batch: tuple[torch.Tensor, ...], stage: RunningStage, stage_str: str) -> dict[str, Any]:
        item_id, loss, logit, label = self.forward(batch)

        self.log(f"{stage_str}/loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=len(label), sync_dist=True, rank_zero_only=True)

        self.losses[stage].append((loss.detach().item(), len(label)))
        self.logits[stage].append(logit.detach().cpu())
        self.labels[stage].append(label.detach().cpu())
        self.indices[stage].append(item_id.detach().cpu())
        return {"loss": loss}

    def end_common(self, stage: RunningStage, stage_str: str) -> None:
        ml = mean_loss(self.losses[stage])
        labels = torch.cat(self.labels[stage])  # [N]
        logits = torch.cat(self.logits[stage])  # [N]
        
        probs = torch.sigmoid(logits)
        metrics = compute_metrics(labels, probs)

        self.log(f"{stage_str}/mean_loss", ml, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log(f"{stage_str}/auroc", metrics['auroc'], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log(f"{stage_str}/auprc", metrics['auprc'], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
        if stage == RunningStage.VALIDATING:
            self.log(self.monitor, metrics['auroc'], logger=False, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)

        self.losses[stage].clear()
        self.logits[stage].clear()
        self.labels[stage].clear()
        self.indices[stage].clear()

    def training_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> dict[str, Any]:
        return self.step_common(batch, RunningStage.TRAINING, 'train')

    def on_train_epoch_end(self) -> None:
        self.end_common(RunningStage.TRAINING, 'train')

    def validation_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> dict[str, Any]:
        return self.step_common(batch, RunningStage.VALIDATING, 'val')

    def on_validation_epoch_end(self) -> None:
        self.end_common(RunningStage.VALIDATING, 'val')

    def test_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int) -> dict[str, Any]:        
        item_id, logit = self.forward(batch)

        self.logits[RunningStage.TESTING].append(logit.detach().cpu())
        self.indices[RunningStage.TESTING].append(item_id.detach().cpu())

    def on_test_epoch_end(self) -> None:
        logits = torch.cat(self.logits[RunningStage.TESTING])  # [N]
        probs = torch.sigmoid(logits)

        indices = torch.cat(self.indices[RunningStage.TESTING])
        df = pd.DataFrame({'id': indices.detach().cpu().numpy(), 'Predicted_Score': probs.detach().cpu().numpy()})
        output_dir = f'{self.cfg.OUTPUT_DIR}/csv_log'
        version_num = len(os.listdir(output_dir)) - 1
        df.to_csv(f'{output_dir}/version_{version_num}/test_logits.csv', index=False)

        self.logits[RunningStage.TESTING].clear()
        self.indices[RunningStage.TESTING].clear()

