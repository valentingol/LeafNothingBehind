"""Cloud models for LNB."""
import abc
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from lnb.architecture.models import Atom


class BaseCloudModel(nn.Module):
    """Base class for cloud model.

    Parameters
    ----------
    base_model : Atom
        Base model to use after cloud processing.
    model_config : Dict | None, optional
        Configuration for the model (without base model).
        By default None (empty).
    """

    def __init__(
        self,
        base_model: Atom,
        model_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.config = model_config
        self.base_model = base_model

    def filter_cloud(self, in_mask_lai_batch: torch.Tensor) -> torch.Tensor:
        """Filter cloudy data and return their indices."""
        cloud_prop = 0.02
        with torch.no_grad():
            # Count number of cloudy/missing pixels in the two time steps
            num_cloud = torch.sum(in_mask_lai_batch[:, :, 0] == 0, dim=(2, 3))
            indices = torch.where(num_cloud > cloud_prop * 256 * 256)
        return indices

    def forward(
        self,
        s1_data: torch.Tensor,
        in_lai: torch.Tensor,
        in_mask_lai: torch.Tensor,
        glob: torch.Tensor,
    ) -> Tuple:
        """Apply cloud filtering and cloud processing before calling base model."""
        with torch.no_grad():
            idx_cloud = self.filter_cloud(in_mask_lai)
            idx_other = (idx_cloud[0], 1 - idx_cloud[1])

        in_lai[idx_cloud], in_mask_lai[idx_cloud] = self.process_cloud(
            lai_cloud=in_lai[idx_cloud],
            lai_other=in_lai[idx_other],
            mask_cloud=in_mask_lai[idx_cloud],
            mask_other=in_mask_lai[idx_other],
        )
        return self.base_model(s1_data, in_lai, in_mask_lai, glob)

    @abc.abstractmethod
    def process_cloud(
        self,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process cloud on batch.

        Parameters
        ----------
        lai_cloud : torch.Tensor
            Cloudy LAI of shape (batch, 1, 256, 256)
        lai_other : torch.Tensor
            Other LAI of shape (batch, 1, 256, 256)
        mask_cloud : torch.Tensor
            Cloud mask of shape (batch, c, 256, 256)
        mask_other : torch.Tensor
            Other mask of shape (batch, c, 256, 256)

        Returns
        -------
        out_lai : torch.Tensor
            De-clouded LAI of shape (batch, 1, 256, 256)
        out_mask : torch.Tensor
            Corresponding mask of shape (batch, c, 256, 256)
        """
        # should return de-clouded lai_cloud
        raise NotImplementedError('You need to implement process_cloud method.')


class HumanCloudModel(BaseCloudModel):
    """Process cloud with human heuristic."""

    def process_cloud(
        self,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process cloud on batch."""
        with torch.no_grad():
            # Normalize lai_other like lai_cloud
            min_cloud = torch.min(lai_cloud, dim=(2, 3), keepdim=True)[0]
            max_cloud = torch.max(lai_cloud, dim=(2, 3), keepdim=True)[0]
            min_other = torch.min(lai_other, dim=(2, 3), keepdim=True)[0]
            max_other = torch.max(lai_other, dim=(2, 3), keepdim=True)[0]
            lai_other_norm = (lai_other - min_other) / (max_other - min_other + 1e-6)
            lai_other = lai_other_norm * (max_cloud - min_cloud + 1e-6) + min_cloud
            out_lai = (mask_cloud[:, 0:1] * lai_cloud
                       + (1 - mask_cloud[:, 0:1]) * lai_other)
            out_mask = mask_cloud
        return out_lai, out_mask
