from pathlib import Path
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage

from coach_pl.configuration import configurable
from coach_pl.dataset import DATASET_REGISTRY
from .neg_sampler import NegSampler

__all__ = ["SLDataset"]


@DATASET_REGISTRY.register()
class SLDataset(Dataset):
    """
    Dataset for SL prediction.
    Each pair of SL genes generates: (cmb_gene_emb, label)
    """
    @configurable
    def __init__(self, stage: RunningStage, cfg: DictConfig) -> None: # sl_root: Path, test_fold: int, cell_line: str | None | float, np_ratio: Optional[int] = None
        self.stage = stage
        # sl_root = cfg.DATASET.SL_ROOT
        # test_fold = cfg.DATASET.TEST_FOLD
        cell_line = cfg.DATASET.CELL_LINE

        if stage == RunningStage.TRAINING:  # VALIDATING, TESTING
            self.df = pd.read_csv(cfg.DATASET.TRAIN_FILE)  #os.path.join(sl_root, f"sl_train_{test_fold}.csv")
            self.df = self.df.sample(frac=1).reset_index(drop=True)  # shuffle
        elif stage == RunningStage.VALIDATING:
            self.df = pd.read_csv(cfg.DATASET.VAL_FILE)  #os.path.join(sl_root, f"sl_test_{test_fold}.csv")
        else:
            self.df = pd.read_csv(cfg.DATASET.TEST_FILE)

        self.cell_line = cell_line
        if cell_line is not None:
            cell_embeddings = np.load('/home/qingyuyang/ESM4SL/data/mapping/clname2embed.npy', allow_pickle=True).item()
            self.cell_embedding = cell_embeddings[cell_line]  # [4079, 6]

    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage) -> dict[str, Any]:
        return {
            "stage": stage,
            "cfg": cfg,
            # "sl_root": cfg.DATASET.SL_ROOT,
            # "test_fold": cfg.DATASET.TEST_FOLD,
            # "cell_line": cfg.DATASET.CELL_LINE
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self):
        """ Subclass should overwrite this method """
        pass

    def pos_neg_ratio(self) -> float:
        pos_indices = self.df[self.df['2'] == 1].index.tolist()
        neg_indices = self.df[self.df['2'] == 0].index.tolist()
        return len(neg_indices) / len(pos_indices)

    @property
    def sampler(self) -> Sampler | None:
        # return None
        return NegSampler(df=self.df, neg_sample_ratio=5) if self.stage == RunningStage.TRAINING and self.pos_neg_ratio() > 5 else None

    @property
    def collate_fn(self) -> Any:
        return None  # Subclass should overwrite this variable if needed
