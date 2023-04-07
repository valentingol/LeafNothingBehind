"""Training functions for Scandium."""
import argparse
import os
from typing import Dict

from time import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb
import yaml

from lnb.architecture.models import Scandium
from lnb.data.dataset import TrainDataset
from lnb.training.metrics import mse_loss


def mask_fn(img_mask):
    """Transform an S2 mask (values between 1 and 9) to float32."""
    metric_mask = np.where(img_mask < 2, 0.0, 1.0)
    metric_mask = np.where(img_mask > 6, 0.0, metric_mask)
    mask_2 = np.where(img_mask == 2, 1.0, 0.0)
    mask_3 = np.where(img_mask == 3, 1.0, 0.0)
    mask_4 = np.where(img_mask == 4, 1.0, 0.0)
    mask_5 = np.where(img_mask == 5, 1.0, 0.0)
    mask_6 = np.where(img_mask == 6, 1.0, 0.0)
    return np.stack([metric_mask, mask_2, mask_3, mask_4, mask_5, mask_6], axis=-1)


def train_loop(config: Dict, model: nn.Module, dataloader: DataLoader,
               device: torch.device) -> None:
    """Training loop."""
    # Save config
    run_id = config['run_id']
    os.makedirs(f'../models/scandium/{run_id}', exist_ok=True)
    with open(f'../models/scandium/{run_id}/config.yaml', 'w', encoding='utf-8') as cfg_file:
        yaml.dump(dict(wandb.config), cfg_file)
    # Get training config params
    train_config = config['train']
    n_epochs = train_config['n_epochs']
    log_interval = train_config['log_interval']
    save_interval = train_config['save_interval']
    learning_rate = train_config['learning_rate']

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    model = model.train()
    n_batch = len(dataloader)
    start_t = time()
    for epoch in range(n_epochs):
        epoch_start_t = time()
        losses = []
        i_batch = 0  # iteration number in current epoch
        for (data, glob) in dataloader:
            i_batch += 1
            # Parse data
            in_lai = data[:, :2, 0:1].to(device)
            target_lai = data[:, 2, 0:1].to(device)
            in_mask_lai = data[:, :2, 1:-2].to(device)
            target_mask_lai = data[:, 2, 1:2].to(device)  # NOTE: binary
            s1_data = data[:, :, -2:].to(device)
            glob = glob.to(device)
            # Forward pass and loss computation
            optimizer.zero_grad()
            pred_lai = model(s1_data=s1_data, in_lai=in_lai,
                             in_mask_lai=in_mask_lai, glob=glob)
            loss = mse_loss(pred_lai, target_lai, target_mask_lai)
            # Backward pass and weight update
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # Logs
            wandb.log({'train_loss': loss.item()})
            if i_batch % log_interval == 0:
                current_t = time()
                eta_ep = int((current_t - epoch_start_t) / i_batch
                             * (n_batch - i_batch))
                eta_ep_str = f"{eta_ep // 3600}h {(eta_ep%3600) // 60}m {eta_ep % 60}s"
                eta = eta_ep + int((current_t - start_t) / (i_batch + epoch * n_batch)
                                   * (n_epochs - epoch - 1) * n_batch)
                eta_str = f"{eta // 3600}h {(eta%3600) // 60}m {eta % 60}s"
                print(f'train loss batch: {loss.item():.4f} - eta epoch {eta_ep_str}'
                      f'- eta {eta_str}     ', end='\r')
        # Epoch logs
        mean_loss = sum(losses) / len(losses)
        wandb.log({'train_loss_mean': mean_loss})
        print(f'\nEpoch {epoch + 1}/{n_epochs}, mean train loss {mean_loss:.4f}')
        if (epoch + 1) % save_interval == 0:
            # Save model
            torch.save(model.state_dict(),
                       f'../models/scandium/{run_id}/{run_id}_ep{epoch+1}.pth')
            print(f'Model saved to ../models/scandium/{run_id}/{run_id}_ep.pth')
    # Save final model
    torch.save(model.state_dict(), f'../models/scandium/{run_id}/{run_id}_last.pth')
    print(f'Model saved to ../models/scandium/{run_id}/{run_id}_last.pth')


def run(config: Dict) -> None:
    """Run training."""
    dataloader = DataLoader(TrainDataset(mask_fn=mask_fn, **config['data']),
                            **config['dataloader'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Scandium(config['model'])
    train_loop(config, model, dataloader, device)


def main() -> None:
    """Main function to run a train with wandb."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=False,
                        default='config/scandium/base.yaml')
    args = parser.parse_args()

    with open(args.config_path, encoding='utf-8') as cfg_file:
        config = yaml.safe_load(cfg_file)
    # New id for model name
    run_id = np.random.randint(1000000)
    config['run_id'] = run_id
    wandb.init(project='lnb', entity='leaf_nothing_behind', group='scandium',
               config=config)
    run(dict(wandb.config))
    wandb.finish()


if __name__ == '__main__':
    main()
