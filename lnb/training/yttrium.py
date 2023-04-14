import argparse
import os
from time import time
from typing import Dict, List, Tuple, Callable
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import wandb
from lnb.architecture.models import Yttrium
from lnb.data.dataset import LNBDataset
from lnb.training.log_utils import get_time_log
from lnb.training.metrics import mse_loss
from lnb.training.trainer import Trainer, parse_data_device, mask_fn

ParsedDataType = Dict[str, Dict[str, torch.Tensor]]

def parse_data_device(data: torch.Tensor, device: torch.device) -> ParsedDataType:
    data, glob, weather_vec = data
    in_lai = data[:, :2, 0:1].to(device)
    lai_target = data[:, 2, 0:1].to(device)
    in_mask_lai = data[:, :2, 1:-2].to(device)
    lai_target_mask = data[:, 2, 1:2].to(device)  # NOTE: binary
    s1_data = data[:, :, -2:].to(device)
    glob = glob.to(device)
    weather_vec = weather_vec.to(device)
    parsed_data = {
        "input_data": {
            "s1_data": s1_data,
            "in_lai": in_lai,
            "in_mask_lai": in_mask_lai,
            "glob": weather_vec,
        },
        "target_data": {"lai_target": lai_target, "lai_target_mask": lai_target_mask},
    }
    return parsed_data

class YttriumTrainer(Trainer):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=False, default="config/yttrium/base.yaml"
    )
    args = parser.parse_args()

    with open(args.config_path, encoding="utf-8") as cfg_file:
        config = yaml.safe_load(cfg_file)

    ########################### Device ###########################

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_built()
        else "cpu"
    )

    ########################### Model ###########################

    model = Yttrium(config["model"])

    ########################### Dataloaders ###########################

    train_dataloader = DataLoader(
        LNBDataset(mask_fn=mask_fn, use_weather=True, **config["data"]), **config["dataloader"]
    )

    # Build validation data loaders
    val_data_config = config["data"].copy()
    val_data_config["grid_augmentation"] = False  # No augmentation for validation

    val_loader_config = config["dataloader"].copy()
    val_loader_config["shuffle"] = False  # No shuffle for validation
    val_loader_config["batch_size"] = 16  # Hard-coded batch size for validation

    val_dataloaders = {}
    for name in ["generalisation", "regular", "mask_cloudy:
        val_data_config["name"] = name
        val_data_config["csv_name"] = f"validation_{name}.csv"
        val_dataloader = DataLoader(
            LNBDataset(mask_fn=mask_fn, use_weather=True, **val_data_config), **val_loader_config
        )
        val_dataloaders["validation_" + name] = val_dataloader

    dataloaders = {
        "train": train_dataloader,
        "validation": val_dataloaders,
    }

    ########################### Train ###########################

    run_id = np.random.randint(1000000)
    config["run_id"] = run_id

    wandb.init(
        project="lnb", entity="leaf_nothing_behind", group="yttrium", config=config
    )

    trainer = YttriumTrainer(
        model=model,
        dataloaders=dataloaders,
        config=config,
        device=device,
        process_func=parse_data_device,
    )
    trainer.run()

    wandb.finish()