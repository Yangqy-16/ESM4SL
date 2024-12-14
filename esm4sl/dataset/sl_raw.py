from pathlib import Path
import random
from typing import Any, Optional
import pickle

import torch
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import RunningStage

from coach_pl.configuration import configurable
from coach_pl.dataset import DATASET_REGISTRY
from .sl import SLDataset

__all__ = ["SLrawDataset"]


class CollateBatch():
    def __init__(self):
        pass

    def __call__(self, batch: list):
        assert type(batch) == list

        outputs = {"gene1": [], "gene2": [], "labels": []}
        for (gene1, seq1), (gene2, seq2), label in batch:
            outputs["gene1"].append((gene1, seq1))
            outputs["gene2"].append((gene2, seq2))
            outputs["labels"].append(label)
        outputs["labels"] = torch.tensor(outputs["labels"])
        return outputs


@DATASET_REGISTRY.register()
class SLrawDataset(SLDataset):
    """
    Dataset for SL prediction.
    Each pair of SL genes generates: (cmb_gene_emb, label)
    """

    @configurable
    def __init__(self, stage: RunningStage, sl_root: Path, cell_line: str, test_fold: int,
                 np_ratio: Optional[int] = None, gene_seq_path: Optional[dict[str, str]] = None) -> None:
        super().__init__(stage, sl_root, cell_line, test_fold, np_ratio)
        with open(gene_seq_path, 'rb') as f:
            self.dic = pickle.load(f)

    @classmethod
    def from_config(cls, cfg: DictConfig, stage: RunningStage) -> dict[str, Any]:
        return {"stage": stage,
                "sl_root": cfg.DATASET.SL_ROOT,
                "cell_line": cfg.DATASET.CELL_LINE,
                "test_fold": cfg.DATASET.TEST_FOLD,
                "np_ratio": cfg.DATASET.NP_RATIO,
                "gene_seq_path": cfg.DATASET.GENE_SEQ_PATH,}

    def __getitem__(self, index: int) -> tuple[tuple[str, str], tuple[str, str], float]:
        # change df's col name if needed
        gene1 = self.df['g1_name'][index]
        gene2 = self.df['g2_name'][index]
        label = float(self.df['label'][index])

        seq1 = self.dic[gene1]
        seq2 = self.dic[gene2]

        if self.stage == RunningStage.TRAINING and random.uniform(0, 1) > 0.5:  # swap 1 and 2 in randomly
            seq1, seq2 = seq2, seq1
            gene1, gene2 = gene2, gene1

        return (gene1, seq1), (gene2, seq2), label

    # For DataLoader
    collate_fn = CollateBatch()
