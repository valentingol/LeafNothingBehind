"""Cloud models for LNB."""
from typing import Tuple

import torch
from torch import nn

from lnb.architecture.models import Atom


class CloudMolecule(nn.Module):
    """Model of two Atom models used for cloud management.

    Parameters
    ----------
    base_model : Atom
        Model to use for non-cloudy data.
    cloud_model : Atom
        Model to use for cloudy data.
    cloud_prop : float, optional
        Threshold of cloud proportion in one input LAI to consider
        the sample as cloudy or not. By default 0.05.
    """

    def __init__(
        self,
        base_model: Atom,
        cloud_model: Atom,
        cloud_prop: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.cloud_model = cloud_model
        self.cloud_prop = cloud_prop
        self.filter_cloud_prop = 0.0

    def filter_cloud(
        self,
        in_mask_lai_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter cloudy data."""
        with torch.no_grad():
            # Filter cloudy data
            clouds = 1 - in_mask_lai_batch[:, 0:1, :, :]  # shape (batch, 2, 256, 256)
            sum_clouds = torch.sum(clouds, dim=(2, 3))  # shape (batch, 2)
            sum_clouds = torch.max(sum_clouds, dim=1)[0]
            idx_cloud = torch.where(sum_clouds >= self.cloud_prop * 256**2)[0]
            idx_nocloud = torch.where(sum_clouds < self.cloud_prop * 256**2)[0]
        self.filter_cloud_prop = idx_cloud.shape[0] / in_mask_lai_batch.shape[0]
        return idx_cloud, idx_nocloud

    def forward(
        self,
        s1_data: torch.Tensor,
        in_lai: torch.Tensor,
        in_mask_lai: torch.Tensor,
        glob: torch.Tensor,
    ) -> Tuple:
        """Apply cloud filtering and cloud processing before calling base model."""
        idx_cloud, idx_nocloud = self.filter_cloud(in_mask_lai)
        out_lai = torch.zeros((in_lai.shape[0], 1, 256, 256), device=in_lai.device)
        # Apply base and cloud model
        if len(idx_cloud) > 0:
            out_lai_cloud = self.base_model(
                s1_data=s1_data[idx_cloud],
                in_lai=in_lai[idx_cloud],
                in_mask_lai=in_mask_lai[idx_cloud],
                glob=glob[idx_cloud],
            )
            out_lai[idx_cloud] = out_lai_cloud
        if len(idx_nocloud) > 0:
            out_lai_nocloud = self.base_model(
                s1_data=s1_data[idx_nocloud],
                in_lai=in_lai[idx_nocloud],
                in_mask_lai=in_mask_lai[idx_nocloud],
                glob=glob[idx_nocloud],
            )
            out_lai[idx_nocloud] = out_lai_nocloud
        return (out_lai,)
