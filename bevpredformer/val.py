import sys
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import pyrootutils
import pytorch_lightning as L
import torch
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.profiler import PyTorchProfiler
from torch.profiler import ProfilerActivity

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from bevpredformer import utils


log = utils.get_pylogger(__name__)
torch.set_float32_matmul_precision("high")

@utils.task_wrapper
def val(cfg: DictConfig) -> Tuple[dict, dict]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    ckpt = utils.get_ckpt_from_path(cfg.ckpt.path)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model = utils.load_state_model(
        model,
        ckpt,
        cfg.ckpt.model.freeze,
        cfg.ckpt.model.load,
        verbose=1,
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    plugins = utils.instantiate_loggers(cfg.get("plugins"))
    if len(plugins) == 0:
        plugins = None

    profiler = None

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        plugins=plugins,
        profiler=profiler,
        accumulate_grad_batches=16//cfg.data.batch_size,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        sys.stderr = open(Path(logger[0].save_dir) / "stdd.err", "a")

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("val"):
        log.info("Starting validation!")
        val_metrics = trainer.validate(model=model, datamodule=datamodule)


    return val_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.modif_config_based_on_flags(cfg)

    utils.extras(cfg)

    # train the model
    metric_dict = val(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
