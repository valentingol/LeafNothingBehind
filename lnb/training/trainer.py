"""Training functions for Scandium."""
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import wandb
import yaml
from logml import Logger
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from lnb.architecture import models as models_module
from lnb.data.dataset import LNBDataset
from lnb.training.metrics import mse_loss

ParsedDataType = Dict[str, Dict[str, torch.Tensor]]


def mask_fn(img_mask: np.ndarray) -> np.ndarray:
    """Process mask function.

    Transform an S2 mask (values between 1 and 9) to float32.
    Channels are last dimension.
    """
    metric_mask = np.where(img_mask < 2, 0.0, 1.0)
    metric_mask = np.where(img_mask > 6, 0.0, metric_mask)
    mask_2 = np.where(img_mask == 2, 1.0, 0.0)
    mask_3 = np.where(img_mask == 3, 1.0, 0.0)
    mask_4 = np.where(img_mask == 4, 1.0, 0.0)
    mask_5 = np.where(img_mask == 5, 1.0, 0.0)
    mask_6 = np.where(img_mask == 6, 1.0, 0.0)
    return np.stack([metric_mask, mask_2, mask_3, mask_4, mask_5, mask_6], axis=-1)


def parse_data_device(
    data: torch.Tensor,
    glob: torch.Tensor,
    device: torch.device,
) -> ParsedDataType:
    """Parse data from dataloader and put it on device."""
    # Parse data
    # Data dims: (batch, time, channels, h, w)
    # Channels:
    #   0: LAI
    #   1: LAI mask (metric)
    #   2 -> -3: LAI mask (other channels)
    #   -2: VV
    #   -1: VH
    in_lai = data[:, :2, 0:1].to(device)
    lai_target = data[:, 2, 0:1].to(device)
    in_mask_lai = data[:, :2, 1:-2].to(device)
    lai_target_mask = data[:, 2, 1:2].to(device)  # NOTE: binary
    s1_data = data[:, :, -2:].to(device)
    glob = glob.to(device)
    parsed_data = {
        "input_data": {
            "s1_data": s1_data,
            "in_lai": in_lai,
            "in_mask_lai": in_mask_lai,
            "glob": glob,
        },
        "target_data": {"lai_target": lai_target, "lai_target_mask": lai_target_mask},
    }
    return parsed_data


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    parsed_data: ParsedDataType,
    **kwargs: Dict,
) -> Tuple:
    """Perform a training step."""
    # Forward pass and loss computation
    optimizer.zero_grad()
    lai_pred, other_pred = model(**parsed_data["input_data"])
    loss = mse_loss(lai_pred=lai_pred, **parsed_data["target_data"])
    if kwargs["interm_supervis"]:
        if kwargs["weight_2"] and kwargs["weight_3"]:
            # Strontium supervisation
            lai_pred2, lai_pred3 = other_pred
            loss2 = mse_loss(
                lai_pred=lai_pred2,
                **parsed_data["target_data"],
            )
            loss3 = mse_loss(
                lai_pred=lai_pred3,
                **parsed_data["target_data"],
            )
            loss_total = loss + kwargs["weight_2"] * loss2 + kwargs["weight_3"] * loss3
        else:
            lai_target_interm = torch.cat(
                [
                    parsed_data["input_data"]["in_lai"],
                    parsed_data["target_data"]["lai_target"].unsqueeze(1),
                ],
                dim=1,
            )
            lai_target_mask_interm = torch.cat(
                [
                    parsed_data["input_data"]["in_mask_lai"][:, :, 0:1],
                    parsed_data["target_data"]["lai_target"].unsqueeze(1),
                ],
                dim=1,
            )
            loss2 = mse_loss(
                lai_pred=other_pred,
                lai_target=lai_target_interm,
                lai_target_mask=lai_target_mask_interm,
            )
            loss_total = loss + loss2
    else:
        loss2 = None
        loss_total = loss
    # Backward pass and weight update
    loss_total.backward()
    optimizer.step()
    return loss, loss2


def valid_step(model: nn.Module, parsed_data: ParsedDataType) -> torch.Tensor:
    """Perform a training step."""
    # Forward pass and loss computation
    with torch.no_grad():
        pred_lai = model(**parsed_data["input_data"])[0]
        loss = mse_loss(pred_lai, **parsed_data["target_data"])
    return loss


# pylint: disable=too-many-locals
def train_val_loop(
    config: Dict,
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloaders: List[DataLoader],
    device: torch.device,
    archi_name: str,
) -> None:
    """Training and validation loop."""
    # Save config
    run_id = config["run_id"]
    os.makedirs(f"../models/{archi_name}/{run_id}", exist_ok=True)
    with open(
        f"../models/{archi_name}/{run_id}/config.yaml",
        "w",
        encoding="utf-8",
    ) as cfg_file:
        yaml.dump(dict(wandb.config), cfg_file)
    # Get training config params
    train_config = config["train"]
    n_epochs = train_config["n_epochs"]
    n_batches = len(train_dataloader)
    lr_decay = train_config["learning_rate_decay"]
    lr_n_mult = train_config["learning_rate_n_mult"]
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    milestones = [int(epoch) for epoch in np.linspace(0, n_epochs, lr_n_mult + 1)][1:]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=lr_decay)

    # Loggers
    train_logger = Logger(
        n_epochs,
        n_batches,
        train_config["log_interval"],
        name='Train',
        name_style='dark_orange3',
        styles='red',
        average=['.*mean.*'],
        bold_keys=True,
    )
    val_loggers = []
    for val_dataloader in val_dataloaders:
        val_name = val_dataloader.dataset.name
        val_logger = Logger(
            n_epochs,
            n_batches=len(val_dataloader),
            name=f"Validation {val_name}",
            name_style='cyan',
            styles='blue',
            average=['valid loss'],
            show_bar=False,
            bold_keys=True,
        )
        val_loggers.append(val_logger)

    # Training and validation loop
    for epoch in range(n_epochs):
        # Training
        model = model.train()
        for data, glob in train_logger.tqdm(train_dataloader):
            # Parse data and put it on device
            parsed_data = parse_data_device(data, glob, device)
            loss, loss2 = train_step(
                model,
                optimizer,
                parsed_data,
                interm_supervis=train_config["interm_supervis"],
                weight_2=train_config["weight_2"],
                weight_3=train_config["weight_3"],
            )
            # Logs
            values = {"train loss": loss.item(), "mean train loss": loss.item()}
            if loss2 is not None:
                values["train loss2"] = loss2.item()
            train_logger.log(values, styles={'train loss2': 'orange3'})
            del values["mean train loss"]
            wandb.log(values, step=train_logger.step)
            wandb.log({"mean train loss": train_logger.mean_vals['train loss']},
                      step=train_logger.step)

        # Validation
        model = model.eval()
        for val_dataloader, val_logger in zip(val_dataloaders, val_loggers):
            # Free unused VRAM
            torch.cuda.empty_cache()
            val_name = val_dataloader.dataset.name
            for data, glob in val_logger.tqdm(val_dataloader):
                # Parse data and put it on device
                parsed_data = parse_data_device(data, glob, device)
                loss = valid_step(model, parsed_data)
                val_logger.log({"valid loss": loss.item()})

            # Epoch validation logs
            mean_valid_loss = val_logger.mean_vals['valid loss']
            wandb.log({f"mean valid loss {val_name}": mean_valid_loss})

        # Free unused VRAM
        torch.cuda.empty_cache()
        # Eventually update learning rate
        scheduler.step()

        # Save model
        if (epoch + 1) % train_config["save_interval"] == 0:
            torch.save(
                model.state_dict(),
                f"../models/{archi_name}/{run_id}/{run_id}_ep{epoch + 1}.pth",
            )
            print(
                f"Model saved to ../models/{archi_name}/{run_id}/"
                f"{run_id}_ep{epoch + 1}.pth",
            )

    # Save final model
    torch.save(model.state_dict(), f"../models/{archi_name}/{run_id}/{run_id}_last.pth")
    print(f"Model saved to ../models/{archi_name}/{run_id}/{run_id}_last.pth")


def run(config: Dict) -> None:
    """Run training."""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_built()
        else "cpu",
    )
    archi_name = config["archi_name"]
    archi_name_cap = "".join([word.capitalize() for word in archi_name.split("_")])
    try:
        model_class = getattr(models_module, archi_name_cap)
    except AttributeError as exc:
        raise ValueError(f"Model {archi_name_cap} not found in "
                         "lnb/architecture/models.py") from exc
    model = model_class(config["model"][archi_name.split('_')[0]]).to(device)
    # Print number of parameters
    n_params = sum(params.numel() for params in model.parameters()
                   if params.requires_grad)
    print(f"Run id {config['run_id']}, architecure {archi_name}")
    print(f"Model has {n_params} trainable parameters")

    train_dataloader = DataLoader(
        LNBDataset(mask_fn=mask_fn, **config["data"]),
        **config["dataloader"],
    )
    # Build validation data loaders
    val_data_config = config["data"].copy()
    val_loader_config = config["dataloader"].copy()
    val_data_config["grid_augmentation"] = False  # No augmentation for validation
    val_loader_config["shuffle"] = False  # No shuffle for validation
    val_loader_config["batch_size"] = config['train']['val_batch_size']
    val_dataloaders = []
    for name in ["regular", "cloudy"]:
        val_data_config["name"] = name
        val_data_config["csv_name"] = f"validation_{name}.csv"
        val_dataloader = DataLoader(
            LNBDataset(mask_fn=mask_fn, **val_data_config),
            **val_loader_config,
        )
        val_dataloaders.append(val_dataloader)
    # Run training
    train_val_loop(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloaders,
        device=device,
        archi_name=config['archi_name'],
    )
