"""Loss and metrics functions."""

import torch


def mse_loss(lai_pred: torch.Tensor, lai_target: torch.Tensor,
             target_lai_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss."""
    return torch.sum(
        target_lai_mask * (lai_pred - lai_target) ** 2
        ) / target_lai_mask.sum()
