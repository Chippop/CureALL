from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from cureall.utils import RankedLogger, extras, instantiate_loggers, log_hyperparameters, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)
rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    # trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    datamodule.prepare_data()
    datamodule.setup()
    labels = []
    for item in datamodule.test_dataloader():
        labels.append(item["label"]["target"])
    labels = torch.cat(labels, dim=0)
    predictions = trainer.predict(model=model, ckpt_path=cfg.ckpt_path, dataloaders=datamodule.test_dataloader())
    predictions = torch.cat(predictions, dim=0)
    predictions = torch.mean(predictions.view(-1, 10, predictions.size(-1)), dim=1)
    labels = torch.mean(labels.view(-1, 10, labels.size(-1)), dim=1)

    with open(cfg.save_path, "wb+") as f:
        import pickle

        pickle.dump((predictions, labels), f)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
