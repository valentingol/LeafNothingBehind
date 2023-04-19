"""Training functions for Scandium."""
import argparse
import os
from time import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import wandb
from lnb.architecture.cloud_models import Altostratus
from lnb.architecture.models import Scandium
from lnb.data.dataset import LNBDataset
from lnb.training.log_utils import get_time_log
from lnb.training.metrics import mse_loss

ParsedDataType = Dict[str, Dict[str, torch.Tensor]]


def mask_fn(img_mask: np.ndarray) -> np.ndarray:
    """Transform an S2 mask (values between 1 and 9) to float32.
    Channels are last dimension."""
    metric_mask = np.where(img_mask < 2, 0.0, 1.0)
    metric_mask = np.where(img_mask > 6, 0.0, metric_mask)
    mask_2 = np.where(img_mask == 2, 1.0, 0.0)
    mask_3 = np.where(img_mask == 3, 1.0, 0.0)
    mask_4 = np.where(img_mask == 4, 1.0, 0.0)
    mask_5 = np.where(img_mask == 5, 1.0, 0.0)
    mask_6 = np.where(img_mask == 6, 1.0, 0.0)
    return np.stack([metric_mask, mask_2, mask_3, mask_4, mask_5, mask_6], axis=-1)


def parse_data_device(
    data: torch.Tensor, glob: torch.Tensor, device: torch.device
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
    interm_supervis: bool,
) -> Tuple:
    """Perform a training step."""
    # Forward pass and loss computation
    optimizer.zero_grad()
    lai_pred, lai_pred2 = model(**parsed_data["input_data"])
    loss = mse_loss(lai_pred=lai_pred, **parsed_data["target_data"])
    if interm_supervis:
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
            lai_pred=lai_pred2,
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


def train_val_loop(
    config: Dict,
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloaders: List[DataLoader],
    device: torch.device,
) -> None:
    """Training and validation loop."""
    # Save config
    run_id = config["run_id"]
    os.makedirs(f"../models/scandium/{run_id}", exist_ok=True)
    with open(
        f"../models/scandium/{run_id}/config.yaml", "w", encoding="utf-8"
    ) as cfg_file:
        yaml.dump(dict(wandb.config), cfg_file)
    # Get training config params
    train_config = config["train"]
    n_epochs = train_config["n_epochs"]
    n_batch = len(train_dataloader)
    lr_decay = train_config["learning_rate_decay"]
    lr_n_mult = train_config["learning_rate_n_mult"]
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    milestones = [int(epoch) for epoch in np.linspace(0, n_epochs, lr_n_mult + 1)][1:]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=lr_decay)

    model = model.to(device)
    start_t = time()
    for epoch in range(n_epochs):
        # Training
        model = model.train()
        epoch_start_t = time()
        train_losses = []
        i_batch = 0  # iteration number in current epoch
        for data, glob in train_dataloader:
            i_batch += 1
            # Parse data and put it on device
            parsed_data = parse_data_device(data, glob, device)
            loss, loss2 = train_step(
                model, optimizer, parsed_data, train_config["interm_supervis"]
            )
            train_losses.append(loss.item())
            # Logs
            wandb.log({"train loss": loss.item()})
            if loss2 is not None:
                wandb.log({"train loss2": loss2.item()})
            if i_batch % train_config["log_interval"] == 0:
                current_t = time()
                eta_str, eta_ep_str = get_time_log(
                    current_t, start_t, epoch_start_t, i_batch, epoch, n_batch, n_epochs
                )
                print(
                    f"train loss batch: {loss.item():.4f} - eta epoch {eta_ep_str}"
                    f"- eta {eta_str}     ",
                    end="\r",
                )

        # Epoch train logs
        mean_train_loss = sum(train_losses) / len(train_losses)
        wandb.log({"mean train loss": mean_train_loss})
        print(f"\nEpoch {epoch + 1}/{n_epochs}, mean train loss {mean_train_loss:.4f}")

        # Validation
        model = model.eval()
        for val_dataloader in val_dataloaders:
            # Free unused VRAM
            torch.cuda.empty_cache()
            val_name = val_dataloader.dataset.name  # type: ignore
            n_batch_val = len(val_dataloader)
            valid_losses = []
            i_batch = 0
            for data, glob in val_dataloader:
                i_batch += 1
                # Parse data and put it on device
                parsed_data = parse_data_device(data, glob, device)
                loss = valid_step(model, parsed_data)
                valid_losses.append(loss.item())
                # Logs
                if i_batch % train_config["log_interval"] == 0:
                    print(
                        f"Validation in progress... batch {i_batch}/{n_batch_val}  ",
                        end="\r",
                    )

            # Epoch validation logs
            mean_valid_loss = sum(valid_losses) / len(valid_losses)
            wandb.log({f"mean valid loss {val_name}": mean_valid_loss})
            print(
                f"\nEpoch {epoch + 1}/{n_epochs}, mean valid loss "
                f"{val_name} {mean_valid_loss:.4f}"
            )

        # Free unused VRAM
        torch.cuda.empty_cache()
        # Eventually update learning rate
        scheduler.step()

        # Save model
        if (epoch + 1) % train_config["save_interval"] == 0:
            torch.save(
                model.state_dict(),
                f"../models/scandium/{run_id}/{run_id}_ep{epoch + 1}.pth",
            )
            print(
                f"Model saved to ../models/scandium/{run_id}/"
                f"{run_id}_ep{epoch + 1}.pth"
            )

    # Save final model
    torch.save(model.state_dict(), f"../models/scandium/{run_id}/{run_id}_last.pth")
    print(f"Model saved to ../models/scandium/{run_id}/{run_id}_last.pth")


def run(config: Dict) -> None:
    """Run training."""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_built()
        else "cpu"
    )
    base_model = Scandium(config["base_model"])
    base_model.load_state_dict(
        torch.load(config['model']['base_model_to_load'], map_location=device),
    )
    # base_model.load_state_dict(
    #     torch.load(
    #         "../models/scandium/clement_best/16062001_last.pth",
    #         map_location=device),
    # )

    # Freeze base model
    for param in base_model.parameters():
        param.requires_grad = False
    # Cloud model
    model = Altostratus(base_model=base_model, model_config=config["model"])
    model = model.to(device)
    # Print number of parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params} parameters")

    train_dataloader = DataLoader(
        LNBDataset(mask_fn=mask_fn, **config["data"]), **config["dataloader"]
    )
    # Build validation data loaders
    val_data_config = config["data"].copy()
    val_loader_config = config["dataloader"].copy()
    val_data_config["grid_augmentation"] = False  # No augmentation for validation
    val_loader_config["shuffle"] = False  # No shuffle for validation
    val_loader_config["batch_size"] = 16  # Hard-coded batch size for validation
    val_dataloaders = []
    for name in ["mask_cloudy"]:
        val_data_config["name"] = name
        val_data_config["csv_name"] = f"validation_{name}.csv"
        val_dataloader = DataLoader(
            LNBDataset(mask_fn=mask_fn, **val_data_config), **val_loader_config
        )
        val_dataloaders.append(val_dataloader)

    # Run training
    train_val_loop(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloaders,
        device=device,
    )


def main() -> None:
    """Main function to run a train with wandb."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=False, default="config/scandium/base.yaml"
    )
    args = parser.parse_args()

    with open(args.config_path, encoding="utf-8") as cfg_file:
        config = yaml.safe_load(cfg_file)
    # New id (for model name)
    run_id = np.random.randint(1000000)
    config["run_id"] = run_id
    wandb.init(
        project="lnb", entity="leaf_nothing_behind", group="scandium_altostratus", config=config
    )
    run(dict(wandb.config))
    wandb.finish()


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    main()
