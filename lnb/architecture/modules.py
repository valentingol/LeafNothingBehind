"""Layers classes and functions."""

from typing import Dict, List, Tuple

import torch
from torch import nn
from transformers import SegformerConfig, SegformerModel


class AutoEncoder(nn.Module):
    """Flexible UNet-like auto-encoder.

    UNet-like auto-encoder with customizable number of layers and channels.
    Input map should be square and have a power of 2 size. Output map has the
    same size as the input map.

    Parameters
    ----------
    in_dim : int
        Number of input channels.
    out_dim : int
        Number of output channels.
    layer_channels : List[int]
        Number of channels for each layer. The length of the list
        is the number of blocks in the encoder and decoder.
    conv_per_layer : int, optional
        Number of convolutional layers per block, by default 1.
    residual : bool, optional
        Whether to use residual connections between the encoder and the decoder.
        By default, False.
    dropout_rate : float, optional
        Dropout rate at each block. By default 0.0 (no dropout).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layer_channels: List[int],
        *,
        conv_per_layer: int = 1,
        residual: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_channels = layer_channels
        self.conv_per_layer = conv_per_layer
        self.residual = residual
        self.dropout_rate = dropout_rate
        # Blocks
        self.encoder_layers = self._build_encoder()
        self.decoder_layers, self.upsample_layers = self._build_decoder()

    def _build_encoder(self) -> nn.ModuleList:
        """Return encoder layers list."""
        channels = [self.in_dim] + self.layer_channels
        encoder_layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            block = nn.Sequential()
            # First layer of each block: dim_i -> dim_i+1
            block.add_module(
                f"encoder_conv_{i + 1}_1",
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
            block.add_module(f"encoder_relu_{i + 1}_{1}", nn.ReLU())
            for j in range(1, self.conv_per_layer):
                # Other layers of each block: dim_i+1 -> dim_i+1
                block.add_module(
                    f"encoder_conv_{i + 1}_{j + 1}",
                    nn.Conv2d(
                        channels[i + 1],
                        channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                block.add_module(f"encoder_relu_{i + 1}_{j + 1}", nn.ReLU())
            # BatchNorm and dropout
            block.add_module(f"encoder_bn_{i + 1}", nn.BatchNorm2d(channels[i + 1]))
            block.add_module(f"encoder_dropout_{i + 1}", nn.Dropout(self.dropout_rate))
            encoder_layers.append(block)
        return encoder_layers

    def _build_decoder(self) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """Return decoder and up-sample layers list."""
        channels = self.layer_channels[::-1] + [self.out_dim]
        decoder_layers = nn.ModuleList()
        upsample_layers = nn.ModuleList()
        for i in range(len(channels) - 2):
            # Up-sample layer
            layer = nn.Sequential()
            layer.add_module(
                f"upsample_{i + 1}",
                nn.ConvTranspose2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
            )
            upsample_layers.append(layer)

            block = nn.Sequential()
            # First layer of each block:
            # 2*dim_i+1 -> dim_i+1 (if residual)
            # dim_i+1 -> dim_i+1 (if not residual)
            if self.residual:
                block.add_module(
                    f"decoder_conv_{i + 1}_1",
                    nn.Conv2d(
                        2 * channels[i + 1],
                        channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
            else:
                block.add_module(
                    f"decoder_conv_{i + 1}_1",
                    nn.Conv2d(
                        channels[i + 1],
                        channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
            block.add_module(f"decoder_relu_{i + 1}_1", nn.ReLU())

            for j in range(1, self.conv_per_layer):
                # Other layers of each block dim_i+1 -> dim_i+1
                block.add_module(
                    f"decoder_conv_{i + 1}_{j + 1}",
                    nn.Conv2d(
                        channels[i + 1],
                        channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                block.add_module(f"decoder_relu_{i + 1}_{j + 1}", nn.ReLU())
            # BatchNorm and dropout
            block.add_module(f"decoder_bn_{i + 1}", nn.BatchNorm2d(channels[i + 1]))
            block.add_module(f"decoder_dropout_{i + 1}", nn.Dropout(self.dropout_rate))
            decoder_layers.append(block)

        # Last layer
        layer = nn.Sequential()
        layer.add_module(
            "final_conv",
            nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=1, padding=0),
        )
        decoder_layers.append(layer)
        return decoder_layers, upsample_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        interm_x = []
        # Encoder
        for i, block in enumerate(self.encoder_layers):
            x = block(x)
            if i < len(self.encoder_layers) - 1:
                interm_x.append(x)
                # Down-sample
                x = nn.MaxPool2d(kernel_size=2, stride=2)(x)
        # Decoder
        for i in range(len(self.decoder_layers) - 1):
            block = self.decoder_layers[i]
            # Up-sample
            x = self.upsample_layers[i](x)
            if self.residual:
                # Concatenate along channel axis
                x = torch.cat([x, interm_x[-i - 1]], dim=1)
            x = block(x)
        # Last layer
        x = self.decoder_layers[-1](x)
        return x


class SegFormerTransposed(nn.Module):
    """SegFormer as encoder and Transposed Conv as decoder.

    Parameters
    ----------
    segformer_config : Dict
        SegFormer configuration.
        num_channels : int
            Number of channels in the input.
        num_encoder_blocks : int
            Number of encoder blocks.
        strides : List[int]
            Strides before each encoder block.
        hidden_sizes : List[int]
            Hidden sizes of each encoder block.
        num_attention_heads : List[int]
            Number of attention heads of each encoder block. The hidden
            sizes of each block should be divisible by the number of
            attention heads in the block.
        hidden_dropout_prob : float
            Dropout rate of the hidden layers.
        attention_probs_dropout_prob : float
            Dropout rate of the attention layers.
    transposed_conv_block_dims : List[int]
        Hidden sizes of each transposed convolution block in the decoder.
    """

    def __init__(
        self,
        segformer_config: Dict,
        transposed_conv_block_dims: List[int],
    ) -> None:
        super().__init__()
        # SegFormer (encoder)
        segformer_hf_config = SegformerConfig(**segformer_config)
        self.segformer = SegformerModel(segformer_hf_config)
        # Transposed Convolutional Block (decoder)
        first_dim = segformer_config["hidden_sizes"][-1]
        conv_block_dims = [first_dim] + transposed_conv_block_dims
        self.conv_block = nn.Sequential()
        for i in range(len(conv_block_dims) - 1):
            self.conv_block.add_module(
                f"trans_conv_{i+1}",
                nn.ConvTranspose2d(
                    conv_block_dims[i],
                    conv_block_dims[i + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
            self.conv_block.add_module(f"trans_conv_{i+1}_relu", nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.segformer(x).last_hidden_state
        x = self.conv_block(x)
        return x


class InceptionBlock(nn.Module):
    """Inception Bock.

    Takes power of 2 input channels and outputs power of 2 output channels.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.branch1x1 = nn.Conv2d(
            in_dim,
            out_dim // 4,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_dim // 2, out_dim // 2, kernel_size=3, stride=1, padding=1),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 8, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_dim // 8, out_dim // 8, kernel_size=5, stride=1, padding=2),
        )
        self.branchpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_dim, out_dim // 8, kernel_size=1, stride=1, padding=0),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x_1 = self.branch1x1(x)
        x_2 = self.branch3x3(x)
        x_3 = self.branch5x5(x)
        x_4 = self.branchpool(x)
        x = torch.cat([x_1, x_2, x_3, x_4], dim=1)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    # Global test
    x = torch.randn(4, 6, 256, 256)  # (batch, channel, height, width)
    model = AutoEncoder(
        in_dim=6,
        out_dim=7,
        layer_channels=[32, 64, 128, 256, 512],
        conv_per_layer=2,
        residual=True,
        dropout_rate=0.2,
    )
    y = model(x)
    print(y.shape)  # (4, 7, 256, 256)
