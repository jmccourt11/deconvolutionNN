"""Reconstruction model for 256x256 diffraction pattern deconvolution."""

from typing import Any, Optional

import torch
import torch.nn as nn

from .base import BaseModel


class ReconModel(BaseModel):
    """
    Reconstruction model for 256x256 diffraction pattern deconvolution.

    This model uses a U-Net-like architecture with skip connections to perform
    deconvolution of diffraction patterns. It's designed specifically for
    256x256 input patterns.

    Attributes:
        encoder1, encoder2, encoder3: Encoder layers with increasing channel depth
        pool: Max pooling layer for downsampling
        drop: Dropout layer for regularization
        bottleneck: Bottleneck layer for feature extraction
        decoder4, decoder3, decoder2: Decoder layers with skip connections
        up_conv4, up_conv3, up_conv2: Upsampling layers
        conv_last: Final output layer with sigmoid activation
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """
        Initialize the ReconModel.

        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)

        def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
            """Create a convolutional block with batch norm and ReLU."""
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=(1, 1)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=(1, 1)),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            return block

        def up_conv(in_channels: int, out_channels: int) -> nn.ConvTranspose2d:
            """Create an upsampling convolutional layer."""
            return nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )

        def conv_last(in_channels: int, out_channels: int) -> nn.Sequential:
            """Create the final convolutional layer with sigmoid activation."""
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=(1, 1)),
                nn.Sigmoid(),
            )
            return block

        nconv = 64
        # Convoluted diffraction pattern encoder
        self.encoder1 = conv_block(1, nconv)
        self.encoder2 = conv_block(nconv, nconv * 2)
        self.encoder3 = conv_block(nconv * 2, nconv * 4)

        self.pool = nn.MaxPool2d((2, 2))
        self.drop = nn.Dropout(0.5)

        self.bottleneck = conv_block(nconv * 4, nconv * 4 * 2)

        # Convoluted diffraction pattern decoder blocks
        self.decoder4 = conv_block(nconv * 4 * 2, nconv * 4)
        self.decoder3 = conv_block(nconv * 4, nconv * 2)
        self.decoder2 = conv_block(nconv * 2, nconv)

        self.up_conv4 = up_conv(nconv * 4 * 2, nconv * 4)
        self.up_conv3 = up_conv(nconv * 4, nconv * 2)
        self.up_conv2 = up_conv(nconv * 2, nconv)
        self.conv_last = conv_last(nconv, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the reconstruction model.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Reconstructed output tensor of shape (B, 1, H, W)
        """
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.drop(self.pool(x1)))
        x3 = self.encoder3(self.drop(self.pool(x2)))

        b = self.bottleneck(self.drop(self.pool(x3)))

        # With skip connections
        d3 = self.up_conv4(b)
        d3 = torch.cat((d3, x3), dim=1)
        d3 = self.decoder4(d3)

        # With skip connections
        d2 = self.up_conv3(d3)
        d2 = torch.cat((d2, x2), dim=1)
        d2 = self.decoder3(d2)

        # With skip connections
        d1 = self.up_conv2(d2)
        d1 = torch.cat((d1, x1), dim=1)
        d1 = self.decoder2(d1)

        d0 = self.conv_last(d1)

        return d0


# Backward compatibility alias
recon_model = ReconModel
