"""Models for the LNB task."""

import abc
from typing import Dict, Tuple

import torch
from einops import rearrange
from torch import nn

from lnb.architecture.modules import AutoEncoder, SegFormerTransposed


class Atom(nn.Module):
    """Base class for LNB models."""

    def __init__(self, model_config: Dict) -> None:
        super().__init__()
        self.config = model_config

    @abc.abstractmethod
    def forward(self, s1_data: torch.Tensor, in_lai: torch.Tensor,
                in_mask_lai: torch.Tensor, glob: torch.Tensor) -> Tuple:
        """Forward pass.

        Parameters
        s1_data : torch.Tensor
            Radar tensor of shape (batch, 3, 2, 256, 256).
            First axis is batch, second is time steps (t-2, t-1, t),
            third is polarization: (VV, VH), then spatial dimensions.
        in_lai : torch.Tensor
            Input LAI tensors of shape (batch, 2, 1, 256, 256).
            First axis is batch, second is time steps (t-2, t-1),
            third is single channel, then spatial dimensions.
        in_mask_lai : torch.Tensor
            Input LAI mask tensors of shape (batch, 2, c, 256, 256).
            First axis is batch, second is time steps (t-2, t-1),
            third is mask channels, then spatial dimensions.
        glob : torch.Tensor
            Global features of shape (batch, n_features).
            Typically the feature that codes the seasonality.

        Returns
        -------
        lai: torch.Tensor
            Output LAI tensor of shape (batch, 1, 256, 256).
        other: torch.Tensor, optional
            Other output tensor (eventually).
        """
        raise NotImplementedError("You must implement the forward pass.")


class Hydrogen(Atom):
    """Hydrogen model for LNB (baseline)."""

    # pylint: disable=unused-argument
    def forward(self, s1_data: torch.Tensor, in_lai: torch.Tensor,
                in_mask_lai: torch.Tensor, glob: torch.Tensor) -> Tuple:
        """Forward pass.

        LAI at t is LAI at t-1 except where mask is 0 where it is LAI at t-2
        (normalized like t-1).

        in_mask_lai must be binary (0 incorrect data).
        """
        lai_0, lai_1 = in_lai[:, 0:1], in_lai[:, 1:]

        mean_0 = lai_0.mean(dim=(2, 3, 4), keepdim=True)
        std_0 = lai_0.std(dim=(2, 3, 4), keepdim=True)
        mean_1 = lai_1.mean(dim=(2, 3, 4), keepdim=True)
        std_1 = lai_1.std(dim=(2, 3, 4), keepdim=True)
        # Normalize LAI at t-2 to N(0, 1)
        new_lai_0 = (lai_0 - mean_0) / (std_0 + 1e-7)
        # Normalize LAI at t-2 like t-1
        new_lai_0 = lai_0 * std_1 + mean_1
        # Compute LAI at t
        lai = lai_1 * in_mask_lai[:, 1] + new_lai_0 * (1 - in_mask_lai[:, 1])
        # Return LAI at t-2 if LAI at t-1 is 0
        for i in range(lai.shape[0]):
            if lai_1[i].sum() == 0:
                lai[i] = lai_0[i]
        return (lai, )


class Scandium(Atom):
    """Scandium model for LNB.

    Parameters
    ----------
    module_config: Dict
        Model configuration.
            s1_ae_config: Dict
                Configuration for the S1 auto-encoder.
                    in_dim: int
                    out_dim : int
                    layer_channels : List[int]
                    conv_per_layer : int, optional
                    residual : bool, optional
                    dropout_rate : float, optional
            mask_module_dim : Tuple[int, int]
                Input and output dimensions of the LAI mask module.
            glob_module_dims : List[int]
                Channels of the global features module.
            conv_block_dims : List[int]
                Channels of the last convolutional block.
    """

    def __init__(self, model_config: Dict) -> None:
        super().__init__(model_config)
        # Config
        s1_ae_config = model_config['s1_ae_config']
        mask_in_dim, mask_out_dim = model_config['mask_module_dim']
        glob_module_dims = model_config['glob_module_dims']
        conv_block_dims = model_config['conv_block_dims']
        # AE for Sentinel-1 data
        self.s1_ae = AutoEncoder(**s1_ae_config)
        # Convolutional layers for LAI mask
        self.conv_lai_mask = nn.Conv2d(mask_in_dim, mask_out_dim,
                                       kernel_size=5, stride=1, padding=2)
        # 1*1 convolutional layers for global features
        self.conv_glob = nn.Sequential()
        for i in range(len(glob_module_dims) - 1):
            self.conv_glob.add_module(
                f"glob_conv_{i+1}",
                nn.Conv2d(glob_module_dims[i], glob_module_dims[i + 1],
                          kernel_size=1, stride=1, padding=0)
                )
            if i < len(glob_module_dims) - 2:
                self.conv_glob.add_module(
                    f"glob_conv_{i+1}_relu",
                    nn.ReLU()
                    )
        # Convolutional block
        first_dim = (mask_out_dim * 2 + glob_module_dims[-1]
                     + s1_ae_config['out_dim'] * 3 + 2)
        conv_block_dims = [first_dim] + conv_block_dims
        self.conv_block = nn.Sequential()
        for i in range(len(conv_block_dims) - 1):
            if i == 0:
                in_channels = conv_block_dims[i]
            else:
                in_channels = conv_block_dims[i] + 2
            self.conv_block.add_module(
                f"conv_{i+1}",
                nn.Conv2d(in_channels, conv_block_dims[i + 1],
                          kernel_size=3, stride=1, padding=1)
                )
            self.conv_block.add_module(
                f"conv_{i+1}_relu",
                nn.ReLU()
                )
        # Last convolutional layer
        self.last_conv = nn.Sequential()
        self.last_conv.add_module(
            f"conv_{len(conv_block_dims)}",
            nn.Conv2d(conv_block_dims[-1] + 2, 1,
                      kernel_size=1, stride=1, padding=0)
        )

    def forward(self, s1_data: torch.Tensor, in_lai: torch.Tensor,
                in_mask_lai: torch.Tensor, glob: torch.Tensor) -> Tuple:
        """Forward pass."""
        batch_size = in_lai.shape[0]
        size = in_lai.shape[-2:]
        # S1 data embedding
        s1_data = rearrange(s1_data, "batch t c h w -> (batch t) c h w")
        s1_embed = self.s1_ae(s1_data)
        s1_embed = rearrange(s1_embed, "(batch t) c h w -> batch t c h w",
                             batch=batch_size)
        # Time steps information embedding
        in_mask_lai = rearrange(in_mask_lai, "batch t c h w -> (batch t) c h w")
        mask_lai_embed = self.conv_lai_mask(in_mask_lai)  # (batch*t, c, h, w)
        mask_lai_embed = rearrange(mask_lai_embed, "(batch t) c h w -> batch (t c) h w",
                                   batch=batch_size)
        s1_input = rearrange(s1_embed[:, :2], "batch t c h w -> batch (t c) h w")
        in_lai = torch.squeeze(in_lai, dim=2)  # (batch, c, h, w)
        t_input = torch.cat([in_lai, mask_lai_embed, s1_input],
                            dim=1)  # (batch, c, h, w)
        # Global features embedding
        glob = rearrange(glob, "batch c -> batch c 1 1")
        glob = glob.repeat(1, 1, size[0], size[1])  # (batch, c, h, w)
        glob = self.conv_glob(glob)  # (batch, c, h, w)

        # Final convolutional block
        x = torch.cat([t_input, glob, s1_embed[:, 2]], dim=1)  # (batch, c, h, w)
        for layer in self.conv_block:
            x = layer(x)
            if layer._get_name() == 'ReLU':  # pylint: disable=protected-access
                # Residual connection with LAI at t-1 and t-2
                x = torch.cat([x, in_lai], dim=1)
        lai = self.last_conv(x)  # (batch, 1, h, w)

        return (lai, s1_embed)  # return s1 embedding for intermediate supervision


class Titanium(Atom):
    """Titanium model for LNB.

    Parameters
    ----------
    module_config: Dict
        Model configuration.
            s1_ae_config: Dict
                Configuration for the S1 auto-encoder.
                    in_dim: int
                    out_dim : int
                    layer_channels : List[int]
                    conv_per_layer : int, optional
                    residual : bool, optional
                    dropout_rate : float, optional
            mask_module_dim : Tuple[int, int]
                Input and output dimensions of the LAI mask module.
            glob_module_dims : List[int]
                Channels of the global features module.
            end_ae_config : Dict
                Configuration for the ending auto-encoder.
                Same parameters as s1_ae_config except in_dim
                and out_dim (not used because automatically inferred).
    """

    def __init__(self, model_config: Dict) -> None:
        super().__init__(model_config)
        # Config
        s1_ae_config = model_config['s1_ae_config']
        mask_in_dim, mask_out_dim = model_config['mask_module_dim']
        glob_module_dims = model_config['glob_module_dims']
        end_ae_config = model_config['end_ae_config']
        # AE for Sentinel-1 data
        self.s1_ae = AutoEncoder(**s1_ae_config)
        # Convolutional layers for LAI mask
        self.conv_lai_mask = nn.Conv2d(mask_in_dim, mask_out_dim,
                                       kernel_size=5, stride=1, padding=2)
        # 1*1 convolutional layers for global features
        self.conv_glob = nn.Sequential()
        for i in range(len(glob_module_dims) - 1):
            self.conv_glob.add_module(
                f"glob_conv_{i+1}",
                nn.Conv2d(glob_module_dims[i], glob_module_dims[i + 1],
                          kernel_size=1, stride=1, padding=0)
                )
            if i < len(glob_module_dims) - 2:
                self.conv_glob.add_module(
                    f"glob_conv_{i+1}_relu",
                    nn.ReLU()
                    )
        # Ending AE
        first_dim = (mask_out_dim * 2 + glob_module_dims[-1]
                     + s1_ae_config['out_dim'] * 3 + 2)
        end_ae_config['in_dim'] = first_dim
        end_ae_config['out_dim'] = 1  # LAI
        self.end_ae = AutoEncoder(**end_ae_config)

    def forward(self, s1_data: torch.Tensor, in_lai: torch.Tensor,
                in_mask_lai: torch.Tensor, glob: torch.Tensor) -> Tuple:
        """Forward pass."""
        batch_size = in_lai.shape[0]
        size = in_lai.shape[-2:]
        # S1 data embedding
        s1_data = rearrange(s1_data, "batch t c h w -> (batch t) c h w")
        s1_embed = self.s1_ae(s1_data)
        s1_embed = rearrange(s1_embed, "(batch t) c h w -> batch t c h w",
                             batch=batch_size)
        # Time steps information embedding
        in_mask_lai = rearrange(in_mask_lai, "batch t c h w -> (batch t) c h w")
        mask_lai_embed = self.conv_lai_mask(in_mask_lai)  # (batch*t, c, h, w)
        mask_lai_embed = rearrange(mask_lai_embed, "(batch t) c h w -> batch (t c) h w",
                                   batch=batch_size)
        s1_input = rearrange(s1_embed[:, :2], "batch t c h w -> batch (t c) h w")
        in_lai = torch.squeeze(in_lai, dim=2)  # (batch, c, h, w)
        t_input = torch.cat([in_lai, mask_lai_embed, s1_input],
                            dim=1)  # (batch, c, h, w)
        # Global features embedding
        glob = rearrange(glob, "batch c -> batch c 1 1")
        glob = glob.repeat(1, 1, size[0], size[1])  # (batch, c, h, w)
        glob = self.conv_glob(glob)  # (batch, c, h, w)

        # Final convolutional block
        x = torch.cat([t_input, glob, s1_embed[:, 2]], dim=1)  # (batch, c, h, w)
        lai = self.end_ae(x)  # (batch, 1, h, w)

        return (lai, s1_embed)  # return s1 embedding for intermediate supervision


class Vanadium(Atom):
    """Vanadium model for LNB.

    Parameters
    ----------
    module_config: Dict
        Model configuration.
            s1_ae_config: Dict
                Configuration for the S1 auto-encoder.
                    in_dim: int
                    out_dim : int
                    layer_channels : List[int]
                    conv_per_layer : int, optional
                    residual : bool, optional
                    dropout_rate : float, optional
            mask_module_dim : Tuple[int, int]
                Input and output dimensions of the LAI mask module.
            glob_module_dims : List[int]
                Channels of the global features module.
            segformer_config : Dict
                Configuration for the SegFormer.
                    num_encoder_blocks : int
                    strides : List[int]
                    hidden_sizes : List[int]
                    num_attention_heads : List[int]
                    hidden_dropout_prob : float
                    attention_probs_dropout_prob : float
            transposed_conv_block_dims : List[int]
                Channels of the transposed convolutional block.
    """

    def __init__(self, model_config: Dict) -> None:
        super().__init__(model_config)
        # Config
        s1_ae_config = model_config['s1_ae_config']
        mask_in_dim, mask_out_dim = model_config['mask_module_dim']
        glob_module_dims = model_config['glob_module_dims']
        segformer_config = model_config['segformer_config']
        conv_block_dims = model_config['transposed_conv_block_dims']
        # AE for Sentinel-1 data
        self.s1_ae = AutoEncoder(**s1_ae_config)
        # Convolutional layers for LAI mask
        self.conv_lai_mask = nn.Conv2d(mask_in_dim, mask_out_dim,
                                       kernel_size=5, stride=1, padding=2)
        # 1*1 convolutional layers for global features
        self.conv_glob = nn.Sequential()
        for i in range(len(glob_module_dims) - 1):
            self.conv_glob.add_module(
                f"glob_conv_{i+1}",
                nn.Conv2d(glob_module_dims[i], glob_module_dims[i + 1],
                          kernel_size=1, stride=1, padding=0)
                )
            if i < len(glob_module_dims) - 2:
                self.conv_glob.add_module(
                    f"glob_conv_{i+1}_relu",
                    nn.ReLU()
                    )
        # SegFormer + Transposed convolutional block
        first_dim = (mask_out_dim * 2 + glob_module_dims[-1]
                     + s1_ae_config['out_dim'] * 3 + 2)
        segformer_config['num_channels'] = first_dim
        self.segformer = SegFormerTransposed(segformer_config=segformer_config,
                                             transposed_conv_block_dims=conv_block_dims)
        # Last convolutional layer
        self.last_conv = nn.Sequential()
        self.last_conv.add_module(
            "conv_last",
            nn.Conv2d(conv_block_dims[-1] + 2, 1,
                      kernel_size=1, stride=1, padding=0)
        )

    def forward(self, s1_data: torch.Tensor, in_lai: torch.Tensor,
                in_mask_lai: torch.Tensor, glob: torch.Tensor) -> Tuple:
        """Forward pass."""
        batch_size = in_lai.shape[0]
        size = in_lai.shape[-2:]
        # S1 data embedding
        s1_data = rearrange(s1_data, "batch t c h w -> (batch t) c h w")
        s1_embed = self.s1_ae(s1_data)
        s1_embed = rearrange(s1_embed, "(batch t) c h w -> batch t c h w",
                             batch=batch_size)
        # Time steps information embedding
        in_mask_lai = rearrange(in_mask_lai, "batch t c h w -> (batch t) c h w")
        mask_lai_embed = self.conv_lai_mask(in_mask_lai)  # (batch*t, c, h, w)
        mask_lai_embed = rearrange(mask_lai_embed, "(batch t) c h w -> batch (t c) h w",
                                   batch=batch_size)
        s1_input = rearrange(s1_embed[:, :2], "batch t c h w -> batch (t c) h w")
        in_lai = torch.squeeze(in_lai, dim=2)  # (batch, c, h, w)
        t_input = torch.cat([in_lai, mask_lai_embed, s1_input],
                            dim=1)  # (batch, c, h, w)
        # Global features embedding
        glob = rearrange(glob, "batch c -> batch c 1 1")
        glob = glob.repeat(1, 1, size[0], size[1])  # (batch, c, h, w)
        glob = self.conv_glob(glob)  # (batch, c, h, w)

        # Final convolutional block
        x = torch.cat([t_input, glob, s1_embed[:, 2]], dim=1)  # (batch, c, h, w)
        x = self.segformer(x)  # (batch, c, h, w)

        # Final convolutional block
        x = torch.cat([x, in_lai], dim=1)  # (batch, c+2, h, w)
        lai = self.last_conv(x)  # (batch, 1, h, w)

        return (lai, s1_embed)  # return s1 embedding for intermediate supervision


class Lithium(Atom):
    def __init__(self, model_config: Dict) -> None:
        super().__init__(model_config)
        ae_config = model_config['ae_config']
        self.ae = AutoEncoder(**ae_config)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)

    
    def forward(self, s1_data: torch.Tensor, in_lai: torch.Tensor, in_mask_lai: torch.Tensor, glob: torch.Tensor) -> Tuple:
        s1_concat = s1_data.view(-1, 6, s1_data.shape[-2], s1_data.shape[-1])

        return self.ae(s1_concat), None
    

class Berylium(Atom):
    def __init__(self, model_config: Dict)-> None:
        super().__init__(model_config)
        ae_config = model_config['ae_config']
        self.ae = AutoEncoder(**ae_config)
        self.mask_conv = nn.Conv2d(1, model_config["mask_embedding_channels"], kernel_size=1, stride=1)

    def forward(self, s1_data: torch.Tensor, in_lai: torch.Tensor, in_mask_lai: torch.Tensor, glob: torch.Tensor) -> Tuple:
        s1_concat = s1_data.view(-1, 6, s1_data.shape[-2], s1_data.shape[-1])
        mask_t2 = self.mask_conv(in_mask_lai[:, 0, :, :, :])
        mask_t1 = self.mask_conv(in_mask_lai[:, 1, :, :, :])

        concat_all = torch.cat([s1_concat, mask_t1, mask_t2], dim=1)

        return self.ae(concat_all), None