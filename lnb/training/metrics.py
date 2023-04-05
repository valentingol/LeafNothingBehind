"""Loss and metrics functions."""

import torch

def mse_loss(lai_pred: torch.Tensor, lai_target: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss."""
    return torch.sum(mask * (lai_pred - lai_target) ** 2) / mask.sum()
