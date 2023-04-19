"""Cloud models for LNB."""
import abc
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import copy

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
        self.filter_prop = 0.0

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
            self.filter_prop = len(idx_recoverable[0]) / (2 * len(in_mask_lai_batch))
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
            s1_data_lai=s1_data[idx_cloud],
            s1_data_other=s1_data[idx_other],
            lai_cloud=in_lai[idx_cloud],
            lai_other=in_lai[idx_other],
            mask_cloud=in_mask_lai[idx_cloud],
            mask_other=in_mask_lai[idx_other],
        )
        return self.base_model(s1_data, in_lai, in_mask_lai, glob)

    def _build_block(
        self,
        channels: List[int],
        kernels: List[int],
    ) -> nn.Module:
        """Return encoder layers list."""
        block = nn.Sequential()
        for i in range(len(channels) - 1):
            block.add_module(
                f'conv{i}',
                nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernels[i],
                    padding='same',
                ))
            if i < len(channels) - 2:
                block.add_module('relu', nn.ReLU())
        return block

    @abc.abstractmethod
    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
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

    # pylint: disable=unused-argument
    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,  # noqa: ARG002
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

            out_mask = (mask_cloud[:, 0:1] * mask_cloud
                        + (1 - mask_cloud[:, 0:1]) * mask_other)

        return out_lai, out_mask


class MlCloudModel_basic(BaseCloudModel):
    def __init__(
        self,
        base_model: Atom,
        model_config: Optional[Dict] = None,
    ) -> None:
        super().__init__(base_model=base_model, model_config=model_config)
        if model_config is None:
            raise ValueError("model_config is required for MlCloudModel.")

        # LAI branch
        self.cloud_mask_layer = nn.Conv2d(
            in_channels=6,
            padding='same',
            **model_config['mask_layer'],
        )
        self.other_mask_layer = nn.Conv2d(
            in_channels=6,
            padding='same',
            **model_config['mask_layer'],
        )
        # Dimension of the LAI + mask_embedding concatenation before LAI conv block
        in_lai_dim = 1 + 1 + (2 * 2)
        self.conv_block_lai = self._build_block(
            channels=[in_lai_dim] + model_config["conv_block_lai"]["channels"],
            kernels=model_config["conv_block_lai"]["kernel_sizes"],
        )

        # Mask branch
        # Dimension of the mask concatenation before mask conv block
        in_mask_dim = 2 * 6
        self.conv_block_mask = self._build_block(
            channels=[in_mask_dim] + model_config["conv_block_mask"]["channels"],
            kernels=model_config["conv_block_mask"]["kernel_sizes"],
        )

    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # LAI branch
        mask_cloud_emb = self.cloud_mask_layer(mask_cloud)
        mask_other_emb = self.other_mask_layer(mask_other)
        input1 = torch.cat([lai_cloud, mask_cloud_emb, lai_other, mask_other_emb],
                           dim=1)
        out_lai = self.conv_block_lai(input1)
        # Mask branch
        input2 = torch.cat([mask_cloud, mask_other], dim=1)
        out_mask = self.conv_block_mask(input2)

        return out_lai, out_mask


class MlCloudModel(MlCloudModel_basic):
    """Full ML module for cloud removal on t, given t-1 and both masks."""

    def __init__(
        self,
        base_model: Atom,
        model_config: Optional[Dict] = None,
    ) -> None:
        super().__init__(base_model=base_model, model_config=model_config)
        if model_config is None:
            raise ValueError("model_config is required for MlCloudModel.")

        # S1 LAI branch
        self.s1_lai_layer = nn.Conv2d(
            in_channels=2,
            padding='same',
            **model_config['s1_layers'],
        )
        # S1 other branch
        self.s1_other_layer = nn.Conv2d(
            in_channels=2,
            padding='same',
            **model_config['s1_layers'],
        )

    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # LAI branch
        mask_cloud_emb = self.cloud_mask_layer(mask_cloud)
        mask_other_emb = self.other_mask_layer(mask_other)

        s1_data_lai_emb = self.s1_lai_layer(s1_data_lai)
        s1_data_other_emb = self.s1_other_layer(s1_data_other)

        input1 = torch.cat([lai_cloud,
                            mask_cloud_emb,
                            lai_other,
                            mask_other_emb,
                            s1_data_lai_emb,
                            s1_data_other_emb],
                           dim=1)

        lai_de_clouded = self.conv_block_lai(input1)
        # Mask branch
        input2 = torch.cat([mask_cloud, lai_cloud, lai_de_clouded], dim=1)
        mask_de_clouded = self.conv_block_mask(input2)

        return lai_de_clouded, mask_de_clouded  # LAI de-clouded, mask


class MixCloudModel(MlCloudModel):
    """Full ML module for cloud removal on t, given t-1 and both masks."""

    def _manual_embeding(
        self,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            out_lai = (mask_cloud[:, 0:1] * lai_cloud
                       + (1 - mask_cloud[:, 0:1]) * lai_other)

        return out_lai, mask_cloud

    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # LAI branch
        mask_cloud_emb = self.cloud_mask_layer(mask_cloud)
        mask_other_emb = self.other_mask_layer(mask_other)

        s1_data_lai_emb = self.s1_lai_layer(s1_data_lai)
        s1_data_other_emb = self.s1_other_layer(s1_data_other)

        input1 = torch.cat([lai_cloud,
                            mask_cloud_emb,
                            lai_other,
                            mask_other_emb,
                            s1_data_lai_emb,
                            s1_data_other_emb],
                           dim=1)

        lai_de_clouded = self.conv_block_lai(input1)
        # Mask branch
        input2 = torch.cat([mask_cloud, lai_cloud, lai_de_clouded], dim=1)
        mask_de_clouded = self.conv_block_mask(input2)

        return lai_de_clouded, mask_de_clouded  # LAI de-clouded, mask


class Cumulus(MlCloudModel):
    """ML module with manual embbeding for mask for cloud removal on t, given t-1 and both masks."""

    def __init__(
        self,
        base_model: Atom,
        model_config: Optional[Dict] = None,
    ) -> None:
        super().__init__(base_model=base_model, model_config=model_config)
        if model_config is None:
            raise ValueError("model_config is required for MlCloudModel.")

        # S1 LAI branch
        self.s1_lai_layer = nn.Conv2d(
            in_channels=base_model.config["s1_ae_config"]["in_dim"],
            padding='same',
            **model_config['s1_layers'],
        )
        # S1 other branch
        self.s1_other_layer = nn.Conv2d(
            in_channels=base_model.config["s1_ae_config"]["in_dim"],
            padding='same',
            **model_config['s1_layers'],
        )

        # LAI branch
        self.cloud_mask_layer = nn.Conv2d(
            in_channels=base_model.config["mask_module_dim"][0],
            padding='same',
            **model_config['mask_layer'],
        )
        self.other_mask_layer = nn.Conv2d(
            in_channels=base_model.config["mask_module_dim"][0],
            padding='same',
            **model_config['mask_layer'],
        )
        # Dimension of the LAI + mask_embedding concatenation before LAI conv block
        in_lai_dim = (2 + 2 * model_config["mask_layer"]["out_channels"] + 2
                      * model_config["s1_layers"]["out_channels"])
        self.conv_block_lai = self._build_block(
            channels=[in_lai_dim] + model_config["conv_block_lai"]["channels"],
            kernels=model_config["conv_block_lai"]["kernel_sizes"],
        )

        # Mask branch
        # Dimension of the mask concatenation before mask conv block
        in_mask_dim = (base_model.config["mask_module_dim"][0] +
                       2 * model_config["mask_layer"]["out_channels"] +
                       2 * model_config["s1_layers"]["out_channels"])
        self.conv_block_mask = self._build_block(
            channels=[in_mask_dim] + model_config["conv_block_mask"]["channels"],
            kernels=model_config["conv_block_mask"]["kernel_sizes"],
        )

    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # LAI branch
        mask_cloud_emb = self.cloud_mask_layer(mask_cloud)
        mask_other_emb = self.other_mask_layer(mask_other)
        manual_mask_emb = (mask_cloud[:, 0:1] * mask_cloud
                           + (1 - mask_cloud[:, 0:1]) * mask_other)
        only_modified_mask = torch.logical_xor(manual_mask_emb, mask_cloud).float()

        s1_data_lai_emb = self.s1_lai_layer(s1_data_lai)
        s1_data_other_emb = self.s1_other_layer(s1_data_other)

        input1 = torch.cat([lai_cloud,
                            mask_cloud_emb,
                            lai_other,
                            mask_other_emb,
                            s1_data_lai_emb,
                            s1_data_other_emb],
                           dim=1)

        lai_de_clouded = self.conv_block_lai(input1)
        # Mask branch

        input2 = torch.cat([mask_cloud, only_modified_mask,
                           lai_cloud, lai_de_clouded], dim=1)
        mask_de_clouded = self.conv_block_mask(input2)

        return lai_de_clouded, mask_de_clouded  # LAI de-clouded, mask


class CumulusV2(Cumulus):
    def __init__(
        self,
        base_model: Atom,
        model_config: Optional[Dict] = None,
    ) -> None:
        super().__init__(base_model=base_model, model_config=model_config)

        # Mask branch
        # Dimension of the mask concatenation before mask conv block
        in_mask_dim = (base_model.config["mask_module_dim"][0] +
                       model_config["mask_layer"]["out_channels"] + 2)
        self.conv_block_mask = self._build_block(
            channels=[in_mask_dim] + model_config["conv_block_mask"]["channels"],
            kernels=model_config["conv_block_mask"]["kernel_sizes"],
        )

    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # LAI branch
        mask_cloud_emb = self.cloud_mask_layer(mask_cloud)
        mask_other_emb = self.other_mask_layer(mask_other)
        manual_mask_emb = (mask_cloud[:, 0:1] * mask_cloud
                           + (1 - mask_cloud[:, 0:1]) * mask_other)
        mask_only_modified = (torch.logical_xor(manual_mask_emb, mask_cloud)).float()
        mask_only_modified_emb = self.cloud_mask_layer(mask_only_modified)

        s1_data_lai_emb = self.s1_lai_layer(s1_data_lai)
        s1_data_other_emb = self.s1_other_layer(s1_data_other)

        input1 = torch.cat([lai_cloud,
                            mask_cloud_emb,
                            lai_other,
                            mask_other_emb,
                            s1_data_lai_emb,
                            s1_data_other_emb],
                           dim=1)

        lai_de_clouded = self.conv_block_lai(input1)
        # Mask branch

        input2 = torch.cat([mask_cloud, mask_only_modified_emb,
                           lai_cloud, lai_de_clouded], dim=1)
        mask_de_clouded = self.conv_block_mask(input2)

        return lai_de_clouded, mask_de_clouded  # LAI de-clouded, mask


class Cirrus(BaseCloudModel):
    """ML module with manual embbeding for mask for cloud removal on t, given t-1 and both masks."""

    def __init__(
        self,
        base_model: Atom,
        model_config: Optional[Dict] = None,
    ) -> None:
        super().__init__(base_model=base_model, model_config=model_config)
        if model_config is None:
            raise ValueError("model_config is required for MlCloudModel.")

        # S1 LAI branch
        self.s1_lai_layer = nn.Conv2d(
            in_channels=base_model.config["s1_ae_config"]["in_dim"],
            padding='same',
            **model_config['s1_layers'],
        )
        # S1 other branch
        self.s1_other_layer = nn.Conv2d(
            in_channels=base_model.config["s1_ae_config"]["in_dim"],
            padding='same',
            **model_config['s1_layers'],
        )

        # Dimension of the LAI + mask_embedding concatenation before LAI conv block
        in_lai_dim = (2 + base_model.config["mask_module_dim"][0] + 2
                      * model_config["s1_layers"]["out_channels"])
        self.conv_block_lai = self._build_block(
            channels=[in_lai_dim] + model_config["conv_block_lai"]["channels"],
            kernels=model_config["conv_block_lai"]["kernel_sizes"],
        )

        # Mask branch
        # Dimension of the mask concatenation before mask conv block
        in_mask_dim = base_model.config["mask_module_dim"][0] + 2
        self.conv_block_mask = self._build_block(
            channels=[in_mask_dim] + model_config["conv_block_mask"]["channels"],
            kernels=model_config["conv_block_mask"]["kernel_sizes"],
        )

    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # LAI branch
        manual_mask_emb = (mask_cloud[:, 0:1] * mask_cloud
                           + (1 - mask_cloud[:, 0:1]) * mask_other)
        only_modified_mask = torch.logical_xor(manual_mask_emb, mask_cloud).float

        s1_data_lai_emb = self.s1_lai_layer(s1_data_lai)
        s1_data_other_emb = self.s1_other_layer(s1_data_other)

        input1 = torch.cat([lai_cloud,
                            only_modified_mask,
                            lai_other,
                            s1_data_lai_emb,
                            s1_data_other_emb],
                           dim=1)

        lai_de_clouded = self.conv_block_lai(input1)
        # Mask branch

        input2 = torch.cat([mask_cloud, lai_cloud, lai_de_clouded], dim=1)
        mask_de_clouded = self.conv_block_mask(input2)

        return lai_de_clouded, mask_de_clouded  # LAI de-clouded, mask


class Altostratus(BaseCloudModel):
    """ML module with manual embbeding for mask for cloud removal on t, given t-1 and both masks."""

    def __init__(
        self,
        base_model: Atom,
        model_config: Optional[Dict] = None,
    ) -> None:
        super().__init__(base_model=base_model, model_config=model_config)
        if model_config is None:
            raise ValueError("model_config is required for MlCloudModel.")

        # S1 LAI branch
        self.s1_lai_layer = nn.Conv2d(
            in_channels=2,
            padding='same',
            **model_config['s1_layers'],
        )
        # S1 other branch
        self.s1_other_layer = nn.Conv2d(
            in_channels=2,
            padding='same',
            **model_config['s1_layers'],
        )

        # LAI branch
        self.cloud_mask_layer = nn.Conv2d(
            in_channels=6,
            padding='same',
            **model_config['mask_layer'],
        )
        # Dimension of the LAI + mask_embedding concatenation before LAI conv block
        in_lai_dim = 8
        self.conv_block_lai = self._build_block(
            channels=[in_lai_dim] + model_config["conv_block_lai"]["channels"],
            kernels=model_config["conv_block_lai"]["kernel_sizes"],
        )

        # Mask branch
        # Dimension of the mask concatenation before mask conv block
        in_mask_dim = 6 + 2
        self.conv_block_mask = self._build_block(
            channels=[in_mask_dim] + model_config["conv_block_mask"]["channels"],
            kernels=model_config["conv_block_mask"]["kernel_sizes"],
        )

    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # LAI branch
        manual_mask_emb = (mask_cloud[:, 0:1] * mask_cloud
                           + (1 - mask_cloud[:, 0:1]) * mask_other)
        only_modified_mask = torch.logical_xor(manual_mask_emb, mask_cloud).float()
        only_modified_mask_emb = self.cloud_mask_layer(only_modified_mask)

        s1_data_lai_emb = self.s1_lai_layer(s1_data_lai)
        s1_data_other_emb = self.s1_other_layer(s1_data_other)

        input1 = torch.cat([lai_cloud,
                            only_modified_mask_emb,
                            lai_other,
                            s1_data_lai_emb,
                            s1_data_other_emb],
                           dim=1)

        lai_de_clouded = self.conv_block_lai(input1)
        # Mask branch

        input2 = torch.cat([mask_cloud, lai_cloud, lai_de_clouded], dim=1)
        mask_de_clouded = self.conv_block_mask(input2)

        return lai_de_clouded, mask_de_clouded  # LAI de-clouded, mask


class Nimbostratus(BaseCloudModel):
    """ML module with manual embbeding for mask for cloud removal on t, given t-1 and both masks."""

    def __init__(
        self,
        base_model: Atom,
        model_config: Optional[Dict] = None,
    ) -> None:
        super().__init__(base_model=base_model, model_config=model_config)
        if model_config is None:
            raise ValueError("model_config is required for MlCloudModel.")

        # S1 LAI branch
        self.s1_lai_layer = nn.Conv2d(
            in_channels=base_model.config["s1_ae_config"]["in_dim"],
            padding='same',
            **model_config['s1_layers'],
        )
        # S1 other branch
        self.s1_other_layer = nn.Conv2d(
            in_channels=base_model.config["s1_ae_config"]["in_dim"],
            padding='same',
            **model_config['s1_layers'],
        )

        # LAI branch
        self.cloud_mask_layer = nn.Conv2d(
            in_channels=base_model.config["mask_module_dim"][0],
            padding='same',
            **model_config['mask_layer'],
        )
        self.lai_remplaced_layer = nn.Conv2d(
            in_channels=7,
            padding='same',
            out_channels=1,
            kernel_size=3
        )
        # Dimension of the LAI + mask_embedding concatenation before LAI conv block
        in_lai_dim = 8
        self.conv_block_lai = self._build_block(
            channels=[in_lai_dim] + model_config["conv_block_lai"]["channels"],
            kernels=model_config["conv_block_lai"]["kernel_sizes"],
        )

        # Mask branch
        # Dimension of the mask concatenation before mask conv block
        in_mask_dim = base_model.config["mask_module_dim"][0] + 2
        self.conv_block_mask = self._build_block(
            channels=[in_mask_dim] + model_config["conv_block_mask"]["channels"],
            kernels=model_config["conv_block_mask"]["kernel_sizes"],
        )

    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # LAI branch
        manual_mask_emb = (mask_cloud[:, 0:1] * mask_cloud
                           + (1 - mask_cloud[:, 0:1]) * mask_other)
        only_modified_mask = torch.logical_xor(manual_mask_emb, mask_cloud).float()
        only_modified_mask_emb = self.cloud_mask_layer(only_modified_mask)

        s1_data_lai_emb = self.s1_lai_layer(s1_data_lai)
        s1_data_other_emb = self.s1_other_layer(s1_data_other)

        input1 = torch.cat([lai_cloud,
                            only_modified_mask_emb,
                            lai_other,
                            s1_data_lai_emb,
                            s1_data_other_emb],
                           dim=1)

        lai_de_clouded = self.conv_block_lai(input1)

        to_remplace = torch.where(only_modified_mask > 1)

        lai_remplaced = copy.deepcopy(lai_cloud)

        lai_remplaced[to_remplace] = lai_de_clouded[to_remplace]

        lai_remplaced = self.lai_remplaced_layer(
            torch.cat((lai_remplaced, only_modified_mask), dim=1))

        # Mask branch

        input2 = torch.cat([mask_cloud, lai_cloud, lai_de_clouded], dim=1)
        mask_de_clouded = self.conv_block_mask(input2)

        return lai_remplaced, mask_de_clouded  # LAI de-clouded, mask


class Nimbostratusv2(Nimbostratus):
    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # LAI branch
        manual_mask_emb = (mask_cloud[:, 0:1] * mask_cloud
                           + (1 - mask_cloud[:, 0:1]) * mask_other)
        only_modified_mask = torch.logical_xor(manual_mask_emb, mask_cloud).float()
        only_modified_mask_emb = self.cloud_mask_layer(only_modified_mask)

        s1_data_lai_emb = self.s1_lai_layer(s1_data_lai)
        s1_data_other_emb = self.s1_other_layer(s1_data_other)

        input1 = torch.cat([lai_cloud,
                            only_modified_mask_emb,
                            lai_other,
                            s1_data_lai_emb,
                            s1_data_other_emb],
                           dim=1)

        lai_de_clouded = self.conv_block_lai(input1)

        to_remplace = torch.where(only_modified_mask > 1)

        lai_remplaced = copy.deepcopy(lai_cloud)

        lai_remplaced[to_remplace] = lai_de_clouded[to_remplace]

        lai_remplaced = self.lai_remplaced_layer(
            torch.cat((lai_remplaced, only_modified_mask), dim=1))

        # Mask branch

        input2 = torch.cat([mask_cloud, lai_cloud, lai_remplaced], dim=1)
        mask_de_clouded = self.conv_block_mask(input2)

        return lai_remplaced, mask_de_clouded  # LAI de-clouded, mask


class Nimbostratusv3(Nimbostratus):
    def __init__(
        self,
        base_model: Atom,
        model_config: Optional[Dict] = None,
    ) -> None:
        super().__init__(base_model=base_model, model_config=model_config)
        if model_config is None:
            raise ValueError("model_config is required for MlCloudModel.")

        self.lai_remplaced_layer = nn.Conv2d(
            in_channels=1,
            padding='same',
            out_channels=1,
            kernel_size=3
        )

    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # LAI branch
        manual_mask_emb = (mask_cloud[:, 0:1] * mask_cloud
                           + (1 - mask_cloud[:, 0:1]) * mask_other)
        only_modified_mask = torch.logical_xor(manual_mask_emb, mask_cloud).float()
        only_modified_mask_emb = self.cloud_mask_layer(only_modified_mask)

        s1_data_lai_emb = self.s1_lai_layer(s1_data_lai)
        s1_data_other_emb = self.s1_other_layer(s1_data_other)

        input1 = torch.cat([lai_cloud,
                            only_modified_mask_emb,
                            lai_other,
                            s1_data_lai_emb,
                            s1_data_other_emb],
                           dim=1)

        lai_de_clouded = self.conv_block_lai(input1)

        to_remplace = torch.where(only_modified_mask > 1)

        lai_remplaced = copy.deepcopy(lai_cloud)

        lai_remplaced[to_remplace] = lai_de_clouded[to_remplace]

        lai_remplaced = self.lai_remplaced_layer(lai_remplaced)

        # Mask branch

        input2 = torch.cat([mask_cloud, lai_cloud, lai_remplaced], dim=1)
        mask_de_clouded = self.conv_block_mask(input2)

        return lai_remplaced, mask_de_clouded  # LAI de-clouded, mask


class Nimbostratusv4(Nimbostratus):
    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # LAI branch
        manual_mask_emb = (mask_cloud[:, 0:1] * mask_cloud
                           + (1 - mask_cloud[:, 0:1]) * mask_other)
        only_modified_mask = torch.logical_xor(manual_mask_emb, mask_cloud).float()
        only_modified_mask_emb = self.cloud_mask_layer(only_modified_mask)

        s1_data_lai_emb = self.s1_lai_layer(s1_data_lai)
        s1_data_other_emb = self.s1_other_layer(s1_data_other)

        input1 = torch.cat([lai_cloud,
                            only_modified_mask_emb,
                            lai_other,
                            s1_data_lai_emb,
                            s1_data_other_emb],
                           dim=1)

        lai_de_clouded = self.conv_block_lai(input1)

        to_remplace = torch.where(only_modified_mask > 1)

        lai_remplaced = copy.deepcopy(lai_cloud)

        lai_remplaced[to_remplace] = lai_de_clouded[to_remplace]

        # Mask branch

        input2 = torch.cat([mask_cloud, lai_cloud, lai_remplaced], dim=1)
        mask_de_clouded = self.conv_block_mask(input2)

        return lai_remplaced, mask_de_clouded  # LAI de-clouded, mask


class Nimbostratusv5(Nimbostratus):
    def __init__(
        self,
        base_model: Atom,
        model_config: Optional[Dict] = None,
    ) -> None:
        super().__init__(base_model=base_model, model_config=model_config)
        if model_config is None:
            raise ValueError("model_config is required for MlCloudModel.")

        self.lai_remplaced_layer = nn.Conv2d(
            in_channels=1,
            padding='same',
            out_channels=1,
            kernel_size=3
        )

        # Dimension of the LAI + mask_embedding concatenation before LAI conv block
        in_lai_dim = 8
        self.conv_block_lai_2 = self._build_block(
            channels=[in_lai_dim] + model_config["conv_block_lai"]["channels"],
            kernels=model_config["conv_block_lai"]["kernel_sizes"],
        )

    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # LAI branch
        manual_mask_emb = (mask_cloud[:, 0:1] * mask_cloud
                           + (1 - mask_cloud[:, 0:1]) * mask_other)
        only_modified_mask = torch.logical_xor(manual_mask_emb, mask_cloud).float()
        only_modified_mask_emb = self.cloud_mask_layer(only_modified_mask)

        s1_data_lai_emb = self.s1_lai_layer(s1_data_lai)
        s1_data_other_emb = self.s1_other_layer(s1_data_other)

        input1 = torch.cat([lai_cloud,
                            only_modified_mask_emb,
                            lai_other,
                            s1_data_lai_emb,
                            s1_data_other_emb],
                           dim=1)

        lai_de_clouded = self.conv_block_lai(input1)

        to_remplace = torch.where(only_modified_mask > 1)

        lai_remplaced = copy.deepcopy(lai_cloud)

        lai_remplaced[to_remplace] = lai_de_clouded[to_remplace]

        input1 = torch.cat([lai_remplaced,
                            only_modified_mask_emb,
                            lai_other,
                            s1_data_lai_emb,
                            s1_data_other_emb],
                           dim=1)

        lai_de_clouded = self.conv_block_lai_2(input1)

        lai_remplaced[to_remplace] = lai_de_clouded[to_remplace]

        # Mask branch

        input2 = torch.cat([mask_cloud, lai_cloud, lai_remplaced], dim=1)
        mask_de_clouded = self.conv_block_mask(input2)

        return lai_remplaced, mask_de_clouded  # LAI de-clouded, mask


class Nimbostratusv6(Nimbostratusv5):
    def __init__(
        self,
        base_model: Atom,
        model_config: Optional[Dict] = None,
    ) -> None:
        super().__init__(base_model=base_model, model_config=model_config)
        if model_config is None:
            raise ValueError("model_config is required for MlCloudModel.")

        # Dimension of the LAI + mask_embedding concatenation before LAI conv block
        in_lai_dim = 9
        self.conv_block_recursive = self._build_block(
            channels=[in_lai_dim] + model_config["conv_block_lai"]["channels"],
            kernels=model_config["conv_block_lai"]["kernel_sizes"],
        )

    def _recursive_layers(
        self,
        nb_recursion: int,
        s1_data_lai_emb: torch.Tensor,
        s1_data_other_emb: torch.Tensor,
        only_modified_mask_emb: torch.Tensor,
        lai_de_clouded: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
    ):
        input_lai_conv = torch.cat([lai_de_clouded,
                                    lai_cloud,
                                    only_modified_mask_emb,
                                    lai_other,
                                    s1_data_lai_emb,
                                    s1_data_other_emb],
                                   dim=1)

        lai_de_clouded = self.conv_block_recursive(input_lai_conv)

        if nb_recursion > 0:
            return self._recursive_layers(
                nb_recursion - 1,
                s1_data_lai_emb,
                s1_data_other_emb,
                only_modified_mask_emb,
                lai_de_clouded,
                lai_cloud,
                lai_other)
        else:
            return lai_de_clouded

    def process_cloud(
        self,
        s1_data_lai: torch.Tensor,
        s1_data_other: torch.Tensor,
        lai_cloud: torch.Tensor,
        lai_other: torch.Tensor,
        mask_cloud: torch.Tensor,
        mask_other: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # LAI branch
        manual_mask_emb = (mask_cloud[:, 0:1] * mask_cloud
                           + (1 - mask_cloud[:, 0:1]) * mask_other)
        only_modified_mask = torch.logical_xor(manual_mask_emb, mask_cloud).float()
        only_modified_mask_emb = self.cloud_mask_layer(only_modified_mask)

        s1_data_lai_emb = self.s1_lai_layer(s1_data_lai)
        s1_data_other_emb = self.s1_other_layer(s1_data_other)

        input_lai_conv = torch.cat([lai_cloud,
                                    only_modified_mask_emb,
                                    lai_other,
                                    s1_data_lai_emb,
                                    s1_data_other_emb],
                                   dim=1)

        lai_de_clouded = self.conv_block_lai(input_lai_conv)

        self._recursive_layers(
            self.config["conv_block_lai"]["recursions"],
            s1_data_lai_emb,
            s1_data_other_emb,
            only_modified_mask_emb,
            lai_de_clouded,
            lai_cloud,
            lai_other)

        # Mask branch

        input2 = torch.cat([mask_cloud, lai_cloud, lai_de_clouded], dim=1)
        mask_de_clouded = self.conv_block_mask(input2)

        return lai_de_clouded, mask_de_clouded  # LAI de-clouded, mask


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
