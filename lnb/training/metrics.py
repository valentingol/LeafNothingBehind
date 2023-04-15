"""Loss and metrics functions."""

import torch


def mse_loss(
    lai_pred: torch.Tensor, lai_target: torch.Tensor, lai_target_mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error loss."""
    return (
        torch.sum(lai_target_mask * (lai_pred - lai_target) ** 2)
        / lai_target_mask.sum()
    )
