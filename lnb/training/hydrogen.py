"""Validation functions for Hydrogen."""

import argparse
import os
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

import wandb
from lnb.architecture.models import Hydrogen
from lnb.data.dataset import LNBDataset
from lnb.training.metrics import mse_loss

ParsedDataType = Dict[str, Dict[str, torch.Tensor]]


def mask_fn(img_mask: np.ndarray) -> np.ndarray:
    """Transform an S2 mask (values between 1 and 9) to float32 binary.

    It uses the simple filter:
    0, 1, 7, 8, 9 -> 0 (incorrect data)
    other -> 1 (correct data)
    """
    interm = np.where(img_mask < 2, 0.0, 1.0)
    return np.where(img_mask > 6, 0.0, interm)


def parse_data_device(data: torch.Tensor, glob: torch.Tensor,
                      device: torch.device) -> ParsedDataType:
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
        'input_data': {'s1_data': s1_data, 'in_lai': in_lai, 'in_mask_lai': in_mask_lai,
                       'glob': glob},
        'target_data': {'lai_target': lai_target,
                        'lai_target_mask': lai_target_mask}
    }
    return parsed_data


def valid_step(model: nn.Module, parsed_data: ParsedDataType) -> torch.Tensor:
    """Perform a training step."""
    # Forward pass and loss computation
    with torch.no_grad():
        pred_lai = model(**parsed_data['input_data'])[0]
        loss = mse_loss(pred_lai, **parsed_data['target_data'])
    return loss


def val_loop(config: Dict, model: nn.Module, val_dataloaders: List[DataLoader],
             device: torch.device) -> None:
    """Training and validation loop."""
    # Save config
    run_id = config['run_id']
    os.makedirs(f'../models/hydrogen/{run_id}', exist_ok=True)
    with open(f'../models/hydrogen/{run_id}/config.yaml',
              'w', encoding='utf-8') as cfg_file:
        yaml.dump(dict(wandb.config), cfg_file)
    # Get config params
    train_config = config['train']
    n_epochs = train_config['n_epochs']

    model = model.to(device)
    for epoch in range(n_epochs):
        # Validation
        model = model.eval()
        for val_dataloader in val_dataloaders:
            val_name = val_dataloader.dataset.name  # type: ignore
            n_batch_val = len(val_dataloader)
            valid_losses = []
            i_batch = 0
            for (data, glob) in val_dataloader:
                i_batch += 1
                # Parse data and put it on device
                parsed_data = parse_data_device(data, glob, device)
                loss = valid_step(model, parsed_data)
                valid_losses.append(loss.item())
                # Logs
                if i_batch % train_config['log_interval'] == 0:
                    print(f'Validation in progress... batch {i_batch}/{n_batch_val}  ',
                          end='\r')

            # Epoch validation logs
            mean_valid_loss = sum(valid_losses) / len(valid_losses)
            wandb.log({f'mean valid loss {val_name}': mean_valid_loss})
            print(f'\nEpoch {epoch + 1}/{n_epochs}, mean valid loss '
                  f'{val_name} {mean_valid_loss:.4f}')


def run(config: Dict) -> None:
    """Run training."""
    model = Hydrogen(config['model'])
    # Build validation data loaders
    val_data_config = config['data'].copy()
    val_loader_config = config['dataloader'].copy()
    val_data_config['grid_augmentation'] = False  # No augmentation for validation
    val_loader_config['shuffle'] = False  # No shuffle for validation
    val_dataloaders = []
    for name in ['generalisation', 'regular', 's2_difference']:
        val_data_config['name'] = name
        val_data_config['csv_name'] = f'validation_{name}.csv'
        val_dataloader = DataLoader(LNBDataset(mask_fn=mask_fn, **val_data_config),
                                    **val_loader_config)
        val_dataloaders.append(val_dataloader)

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_built() else 'cpu')
    # Run training
    val_loop(config=config, model=model, val_dataloaders=val_dataloaders,
             device=device)


def main() -> None:
    """Main function to run a train with wandb."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=False,
                        default='config/hydrogen/base.yaml')
    args = parser.parse_args()

    with open(args.config_path, encoding='utf-8') as cfg_file:
        config = yaml.safe_load(cfg_file)
    # New id (for model name)
    run_id = np.random.randint(1000000)
    config['run_id'] = run_id
    wandb.init(project='lnb', entity='leaf_nothing_behind', group='hydrogen',
               config=config)
    run(dict(wandb.config))
    wandb.finish()


if __name__ == '__main__':
    main()
