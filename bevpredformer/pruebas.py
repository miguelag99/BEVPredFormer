import os
import time

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import pyrootutils
import lightning as L
import torch
from omegaconf import DictConfig
from lightning.pytorch import LightningDataModule, Trainer, Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.profilers import PyTorchProfiler

import matplotlib.pyplot as plt


from torch.profiler import ProfilerActivity

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from bevpredformer import utils
import pickle

log = utils.get_pylogger(__name__)

def train(cfg: DictConfig) -> Tuple[dict, dict]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    dataloader = datamodule.val_dataloader()
    iter_dataloader = iter(dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model).to(device)
    
    import pdb; pdb.set_trace()
    
    
    return 0, 0
    
@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.modif_config_based_on_flags(cfg)
    
    utils.extras(cfg)

    train(cfg)



if __name__ == "__main__":
    main()
