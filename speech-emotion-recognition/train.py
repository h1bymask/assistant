import random
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import wandb
from core import Learner
from core import get_train_dataloader, get_train_dataset, get_val_dataloader, get_val_dataset, load_jsonl_as_df


@hydra.main(config_path="conf", config_name="config")
def train_model(config):
    with open("wandb_token.txt") as f:
        token = f.readline()
        wandb.login(key=token, relogin=True)
    model_name = config.model["_target_"].split(".")[-1]
    model = instantiate(config.model)
    wandb.init(
        project="speech-emotion-recognition",
        name=datetime.now().strftime("%Y/%m/%d, %H:%M"),
        group=model_name,
        config=OmegaConf.to_container(config),
        config_exclude_keys=[
            "train_manifest_path",
            "val_manifest_path",
            "base_path",
            "hydra",
            "collate_fn",
            "augm_func",
            "get_train_weights",
        ],
    )
    wandb.watch(models=model, log="parameters")
    train_manifest_path = Path(config.train_manifest_path)
    train_dataset = get_train_dataset(
        load_jsonl_as_df(train_manifest_path),
        ds_base_path=train_manifest_path.parent,
        get_train_weights=config.get_train_weights,
        augm_func=instantiate(config.augm_func),
    )
    val_manifest_path = Path(config.val_manifest_path)
    val_dataset = get_val_dataset(load_jsonl_as_df(val_manifest_path), ds_base_path=val_manifest_path.parent)
    dataloaders = {
        "train": get_train_dataloader(
            train_ds=train_dataset,
            batch_size=config.batch_size,
            collate_fn=instantiate(config.collate_fn, _partial_=True),
        ),
        "val": get_val_dataloader(
            val_ds=val_dataset,
            batch_size=3*config.batch_size,
            collate_fn=instantiate(config.collate_fn, _partial_=True),
        ),
    }

    # load pretrained model
    if config.pt_model_path is not None:
        model.load_state_dict(torch.load(config.pt_model_path, map_location="cuda:0"))

    # init learner
    learner = Learner(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dataloaders=dataloaders,
        model=model, 
        batch_size=config.batch_size,
    )

    # train
    _, counts = np.unique(train_dataset.df.label.values, return_counts=True)
    class_weights = torch.FloatTensor(counts/sum(counts)).to(learner.device)
    best_model_wts = learner.train(**config.train_params, loss_type='focal', alpha=0.1/class_weights, gamma=2)

    # dump best model
    model_folder = Path(config.best_model_folder)
    model_folder.mkdir(exist_ok=True)

    torch.save(best_model_wts, model_folder / model_name)


if __name__ == "__main__":
    # # fix seeds for reproducibility
    # torch.manual_seed(0)
    # random.seed(0)
    # np.random.seed(0)
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    train_model()
