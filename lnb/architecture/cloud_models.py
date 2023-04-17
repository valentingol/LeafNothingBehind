"""Cloud models for LNB."""
import abc
from typing import Dict, List, Optional, Tuple

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

    def filter_cloud(self, in_mask_lai_batch: torch.Tensor) -> List[torch.Tensor]:
        """Filter cloudy data so that de-clouding is possible with other time step."""
        cloud_prop = 0.02
        with torch.no_grad():
            # Count number of cloudy/missing pixels (= mask channel 0) that are
            # not cloudy/missing in the other time step
            # Operation considered is [A AND (A XOR B)] so that:
            # (A=True, B=False) -> True, rest -> False
            clouds = (in_mask_lai_batch[:, :, 0] == 0)  # (batch, 2, 256, 256)
            clouds_xor = torch.logical_xor(clouds[:, 0],
                                           clouds[:, 1])  # (batch, 256, 256)
            clouds_recoverable = torch.cat([
                torch.unsqueeze(clouds[:, 0] * clouds_xor, dim=1),
                torch.unsqueeze(clouds[:, 1] * clouds_xor, dim=1),
            ], dim=1)  # (batch, 2, 256, 256)
            idx_recoverable = torch.where(
                torch.sum(clouds_recoverable, dim=(2, 3)) > cloud_prop * 256**2,
            )
        return idx_recoverable

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
        # should return de-clouded lai_cloud and corresponding mask
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
            cloud_mean = torch.mean(lai_cloud, dim=(2, 3), keepdim=True)
            cloud_std = torch.std(lai_cloud, dim=(2, 3), keepdim=True)
            other_mean = torch.mean(lai_other, dim=(2, 3), keepdim=True)
            other_std = torch.std(lai_other, dim=(2, 3), keepdim=True)
            lai_other = (lai_other - other_mean) / (other_std + 1e-6)
            lai_other = lai_other * cloud_std + cloud_mean

            out_lai = (mask_cloud[:, 0:1] * lai_cloud
                       + (1 - mask_cloud[:, 0:1]) * lai_other)
            out_mask = mask_cloud
        return out_lai, out_mask


class MlCloudModel(BaseCloudModel):
    """Full ML module for cloud removal on t, given t-1 and both masks"""

    def __init__(
            self,
            cr_config: Dict,
            model_config: Dict,
            base_model) -> None:
        super().__init__(base_model=base_model, model_config=model_config)
        # Blocks
        self.mask_layers = self._build_conv(cr_config["mask_conv_config"], "mask_conv")
        self.other_mask_layers = self._build_conv(
            cr_config["other_mask_config"], "other_mask")
        self.cr1_layers = self._build_conv(cr_config["cr1_config"], "cr_1")
        self.cr2_layers = self._build_conv(cr_config["cr2_config"], "cr_2")

    def _build_conv(self, conv_config, name) -> nn.ModuleList:
        """Return encoder layers list."""
        channels = conv_config["conv"]
        encoder_layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            block = nn.Sequential()
            block.add_module(
                f"{name}_conv_{i + 1}_1",
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=conv_config["kernel"],
                    stride=1,
                    padding="same",
                ),
            )
            block.add_module(f"{name}_relu_{i + 1}_{1}", nn.ReLU())
            # BatchNorm and dropout
            block.add_module(f"{name}_bn_{i + 1}", nn.BatchNorm2d(channels[i + 1]))
            block.add_module(
                f"{name}_dropout_{i + 1}",
                nn.Dropout(
                    conv_config["dropout_rate"]))
            encoder_layers.append(block)
        return encoder_layers

    def process_cloud(self, lai_cloud: torch.Tensor, lai_other: torch.Tensor,
                      mask_cloud: torch.Tensor, mask_other: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # lai_mask_layers
        for block in self.lai_mask_layers:
            mask_cloud = block(mask_cloud)

        # substitute_lai_mask_layers
        for block in self.substitute_lai_mask_layers:
            mask_other = block(mask_other)

        x = torch.cat([lai_cloud, mask_cloud, lai_other, mask_other], dim=2)
        # conv_1_layers
        for block in self.conv_1_layers:
            x = block(x)

        x = torch.cat([x, lai_cloud], dim=1)
        # conv_2_layers
        for block in self.conv_2_layers:
            x = block(x)

        return x, lai_cloud  # LAI de-clouded


if __name__ == '__main__':
    class BaseModel(nn.Module):
        """Mock base model."""

        # pylint: disable=unused-argument
        def forward(self, *args: Tuple) -> torch.Tensor:  # noqa
            """Forward pass."""
            return torch.empty(0)

    base_model = BaseModel()
    model = HumanCloudModel(base_model=base_model)  # type: ignore
    s1_data = torch.rand(8, 3, 2, 256, 256)
    in_lai = torch.rand(8, 2, 1, 256, 256)
    in_mask_lai = torch.randint(0, 48, (8, 2, 6, 256, 256))
    glob = torch.rand(8, 2)
    model(s1_data, in_lai, in_mask_lai, glob)
