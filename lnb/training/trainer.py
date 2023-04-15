"""Generic trainer for LNB models."""
import argparse
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import wandb
from lnb.architecture.models import Scandium
from lnb.data.dataset import LNBDataset
from lnb.training.metrics import mse_loss

ParsedDataType = Dict[str, Dict[str, torch.Tensor]]


def mask_fn(img_mask: np.ndarray) -> np.ndarray:
    """Mask processing function.

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


def parse_data_device(data: torch.Tensor, device: torch.device) -> ParsedDataType:
    """Parse data from dataloader and put it on device.

    Parse data
    Data dims: (batch, time, channels, h, w)
    Channels:
      0: LAI
      1: LAI mask (metric)
      2 -> -3: LAI mask (other channels)
      -2: VV
      -1: VH
    """
    data, glob = data
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


class Trainer:
    """Base trainer class."""

    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        process_func: Callable,
        device: torch.device,
        config: Optional[Dict] = None,
    ) -> None:
        self.model = model.to(device)
        self.dataloders = dataloaders
        self.process_func = process_func
        self.config = config
        self.device = device

        self.optimizer, self.scheduler = self.configure_optimizers()

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure the optimizers and the learning rate schedulers.

        Returns
        -------
        Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]
            Optimizer and scheduler
        """
        train_config = self.config["train"]

        n_epochs = train_config["n_epochs"]
        lr_decay = train_config["learning_rate_decay"]
        lr_n_mult = train_config["learning_rate_n_mult"]

        milestones = [
            int(epoch) for epoch in np.linspace(0, n_epochs, lr_n_mult + 1)
        ][1:]

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=train_config["learning_rate"],
        )
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=lr_decay)
        return optimizer, scheduler

    def train_step(self, parsed_data: ParsedDataType, loss_names: list) -> Tuple:
        """Make a train step.

        Parameters
        ----------
        data ParsedDataType:
            Parsed data. Contains 'input_data' and 'target_data' keys.
        loss_names (list): Loss names

        Returns
        -------
            Tuple: Losses
        """
        self.optimizer.zero_grad()
        predictions = self.model(**parsed_data["input_data"])
        losses = [
            mse_loss(lai_pred=pred, **parsed_data["target_data"])
            for pred in predictions
            if pred is not None
            and all(pred.shape == d.shape for d in parsed_data["target_data"].values())
        ]

        losses[0].backward()
        self.optimizer.step()

        return {
            loss_name: loss
            for loss_name, loss in zip(loss_names, losses)
            if loss is not None
        }

    def validation_step(self, data: ParsedDataType, loss_names: list) -> Tuple:
        """Make a validation step.

        Parameters
        ----------
        data : ParsedDataType
            Parsed data. Contains 'input_data' and 'target_data' keys.
        loss_names : list
            Loss names

        Returns
        -------
            Tuple: Losses
        """
        with torch.no_grad():
            predictions = self.model(**data["input_data"])
            losses = [
                mse_loss(lai_pred=pred, **data["target_data"])
                for pred in predictions
                if pred is not None
                and all(pred.shape == d.shape for d in data["target_data"].values())
            ]

        return {
            loss_name: loss
            for loss_name, loss in zip(loss_names, losses)
            if loss is not None
        }

    def general_step(
        self,
        name: str,
        dataloader: DataLoader,
        progress_bar,
        config: Dict,
        subname: str = None,
    ) -> None:
        """General step which can be used for train step and val steps.

        Args:
            name (str): _description_
            dataloader (DataLoader): _description_
            progress_bar (_type_): _description_
            config (Dict): _description_

        Returns
        -------
            _type_: _description_
        """
        all_losses = defaultdict(list)
        all_mean_losses = defaultdict(list)
        all_std_losses = defaultdict(list)

        pbar = tqdm(
            enumerate(dataloader),
            desc=name.capitalize(),
            total=len(dataloader),
        )
        for idx_batch, data in pbar:
            processed_data = self.process_func(data, self.device)
            losses = getattr(self, f"{name}_step")(
                processed_data, loss_names=["mse_loss"],
            )

            if subname:
                name = f"{name}/{subname}"

            for loss_name, loss_value in losses.items():
                loss_value = loss_value.detach().cpu().numpy()
                # Clip loss value to [0, 1]
                loss_value = np.clip(loss_value, 0, 1)
                all_losses[loss_name].append(loss_value)
                wandb.log({f"{name}/{loss_name}": loss_value})

            pbar.set_postfix(
                {
                    f"{name}/{loss_name}": loss_value
                    for loss_name, loss_value in losses.items()
                },
            )

        # Wandb log mean losses
        for loss_name, loss_values in all_losses.items():
            mean_loss = np.mean(loss_values)
            std_loss = np.std(loss_values)

            all_mean_losses[loss_name] = mean_loss
            all_std_losses[loss_name] = std_loss

            wandb.log({f"{name}/mean_{loss_name}": mean_loss})
            wandb.log({f"{name}/std_{loss_name}": std_loss})

        progress_bar.set_postfix(
            {
                f"{name}/mean_{loss_name}": loss_value
                for loss_name, loss_value in all_mean_losses.items()
            },
        )

        return all_mean_losses, all_std_losses, all_losses

    def run(self):
        run_id = self.config["run_id"]
        train_config = self.config["train"]
        n_epochs = train_config["n_epochs"]

        defaultdict(list)
        defaultdict(list)

        epoch_pbar = trange(n_epochs, desc="Epochs")
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch+1}/{n_epochs}")

            # Train
            self.model.train()
            train_mean_losses, train_std_losses, all_losses = self.general_step(
                "train", self.dataloders["train"], epoch_pbar, train_config,
            )

            # Validation
            self.model.eval()
            for val_name, val_dataloader in self.dataloders["validation"].items():
                torch.cuda.empty_cache()
                train_mean_losses, train_std_losses, all_losses = self.general_step(
                    "validation",
                    val_dataloader,
                    epoch_pbar,
                    None,
                    subname=val_name.split("_")[1],
                )

            torch.cuda.empty_cache()
            self.scheduler.step()

            if (epoch + 1) % train_config["save_interval"] == 0:
                torch.save(
                    model.state_dict(),
                    f"../models/{self.model.__name__.lower()}/{run_id}/{run_id}_ep{epoch + 1}.pth",
                )
                print(
                    f"Model saved to ../models/{self.model.__name__.lower()}/{run_id}/"
                    f"{run_id}_ep{epoch + 1}.pth",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=False, default="config/scandium/base.yaml",
    )
    args = parser.parse_args()

    with open(args.config_path, encoding="utf-8") as cfg_file:
        config = yaml.safe_load(cfg_file)

    # Device

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_built()
        else "cpu",
    )

    # Model

    model = Scandium(config["model"])

    # Data loaders

    train_dataloader = DataLoader(
        LNBDataset(mask_fn=mask_fn, **config["data"]), **config["dataloader"],
    )

    # Build validation data loaders
    val_data_config = config["data"].copy()
    val_data_config["grid_augmentation"] = False  # No augmentation for validation

    val_loader_config = config["dataloader"].copy()
    val_loader_config["shuffle"] = False  # No shuffle for validation
    val_loader_config["batch_size"] = 16  # Hard-coded batch size for validation

    val_dataloaders = {}
    for name in ["generalisation", "regular", "mask_cloudy"]:
        val_data_config["name"] = name
        val_data_config["csv_name"] = f"validation_{name}.csv"
        val_dataloader = DataLoader(
            LNBDataset(mask_fn=mask_fn, **val_data_config), **val_loader_config,
        )
        val_dataloaders["validation_" + name] = val_dataloader

    dataloaders = {
        "train": train_dataloader,
        "validation": val_dataloaders,
    }

    # Train

    run_id = np.random.randint(1000000)
    config["run_id"] = run_id

    wandb.init(
        project="lnb", entity="leaf_nothing_behind", group="strontium", config=config,
    )

    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        config=config,
        device=device,
        process_func=parse_data_device,
    )
    trainer.run()

    wandb.finish()
