from pathlib import Path
import os
from typing import Any

import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, Sampler
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage

from coach_pl.configuration import configurable
from coach_pl.dataset import DATASET_REGISTRY


STAGE_MAPPING = {RunningStage.TRAINING: 'train',
                 RunningStage.VALIDATING: 'val',
                 RunningStage.TESTING: 'test'}


class CollateBatch():
    def __init__(self):
        pass

    def padding(self, genes: list[torch.Tensor], max_len: int = 2000) -> tuple[torch.Tensor, torch.Tensor]:
        count_len = [g.shape[0] for g in genes]

        batched_sample = torch.zeros((len(genes), max_len, genes[0].shape[1]))
        for i, (gene, length) in enumerate(zip(genes, count_len)):
            if length <= max_len:
                batched_sample[i, :length, :] = gene
            else:
                batched_sample[i, :max_len, :] = gene[:max_len, :]
        batched_sample = batched_sample.float()

        prot_mask = torch.ones((len(genes), max_len), dtype=bool)
        for i, seq_len in enumerate(count_len):
            if seq_len <= max_len:
                prot_mask[i, :seq_len] = 0
            else:
                prot_mask[i, :] = 0

        return batched_sample, prot_mask

    def __call__(self, 
        batch: list[tuple[int, torch.Tensor, torch.Tensor, float]] | list[tuple[int, torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, ...]:
        if len(batch[0]) == 4:
            item_ids, g1, g2, labels = zip(*batch)
        else:
            item_ids, g1, g2 = zip(*batch)

        batched_g1, g1_mask = self.padding(g1)
        batched_g2, g2_mask = self.padding(g2)

        if len(batch[0]) == 4:
            return torch.tensor(item_ids), batched_g1, g1_mask, batched_g2, g2_mask, torch.tensor(labels)
        else:
            return torch.tensor(item_ids), batched_g1, g1_mask, batched_g2, g2_mask


@DATASET_REGISTRY.register()
class HW3Dataset(Dataset):
    """
    Dataset for SL prediction.
    Each pair of SL genes generates: (id, gene1_emb, gene2_emb, label)
    """
    @configurable
    def __init__(self, stage: RunningStage, sl_root: Path, esm_root: Path) -> None: #  | float, np_ratio: Optional[int] = None
        self.stage = stage
        self.df = pd.read_csv(os.path.join(sl_root, f"{STAGE_MAPPING[stage]}.csv"))
        self.df = self.df.sample(frac=1).reset_index(drop=True)  # shuffle
        self.esm_root = esm_root

    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage) -> dict[str, Any]:
        return {
            "stage": stage,
            "sl_root": cfg.DATASET.SL_ROOT,
            "esm_root": cfg.DATASET.ESM_ROOT,
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor, torch.Tensor, float]:
        item_id = self.df['id'][index]
        g1_idx = self.df['geneA_ID'][index]
        g2_idx = self.df['geneB_ID'][index]
        if self.stage != RunningStage.TESTING:
            label = float(self.df['label'][index])

        gene1 = np.load(os.path.join(self.esm_root, f"{g1_idx}.npy"), allow_pickle=True).item()['rep']  # [L, 1280], L different across samples!
        gene1 = torch.from_numpy(gene1)
        gene2 = np.load(os.path.join(self.esm_root, f"{g2_idx}.npy"), allow_pickle=True).item()['rep']
        gene2 = torch.from_numpy(gene2)

        if self.stage != RunningStage.TESTING:
            return item_id, gene1, gene2, label
        else:
            return item_id, gene1, gene2

    @property
    def sampler(self) -> Sampler | None:
        return None

    @property
    def collate_fn(self) -> Any:
        return CollateBatch()
