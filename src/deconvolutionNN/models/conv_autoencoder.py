"""Convolutional autoencoder with skip connections for deconvolution."""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel


class ConvAutoencoderSkip(BaseModel):
    """
    Convolutional autoencoder with skip connections for deconvolution.

    This model uses a U-Net-like architecture with skip connections to perform
    deconvolution of diffraction patterns. It includes a probe kernel for
    realistic diffraction pattern generation.

    Attributes:
        enc1, enc2: Encoder layers with increasing channel depth
        pool1, pool2: Max pooling layers for downsampling
        bottleneck: Bottleneck layer for feature extraction
        up1, up2: Upsampling layers
        dec1, dec2: Decoder layers with skip connections
        final_layer: Final output layer with sigmoid activation
        probe_kernel: Registered buffer containing the probe kernel
    """

    def __init__(self, probe_kernel: np.ndarray) -> None:
        """
        Initialize the ConvAutoencoderSkip model.

        Args:
            probe_kernel: Complex probe kernel for diffraction pattern generation
                         Shape should be (H, W) for complex values
        """
        super().__init__()

        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Final output layer
        self.final_layer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.75)

        # Register probe kernel as buffer
        self._register_probe_kernel(probe_kernel)

    def _register_probe_kernel(self, probe_kernel: np.ndarray) -> None:
        """
        Register the probe kernel as a buffer.

        Args:
            probe_kernel: Complex probe kernel array
        """
        probe_tensor = torch.from_numpy(probe_kernel).float()
        # Add batch and channel dimensions: (1, 1, H, W)
        probe_tensor = probe_tensor.unsqueeze(0).unsqueeze(0)
        self.register_buffer("probe_kernel", probe_tensor)

    def conv2d_probe(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D convolution with the probe kernel.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Convolved output tensor of same shape as input
        """
        return F.conv2d(x, self.probe_kernel, padding="same")

    def fft_conv2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply FFT-based convolution for realistic diffraction pattern generation.

        This method performs convolution in the frequency domain to generate
        realistic diffraction patterns from the decoded object.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Diffraction pattern tensor of same shape as input
        """
        # Compute FFT of decoded image
        x_fft = torch.fft.ifft2(x)

        # Multiply probe and object in real space
        real_space_product = x_fft * self.probe_kernel

        # Take FFT of product to get diffraction pattern
        output_fft = torch.fft.fft2(real_space_product)

        # Take magnitude squared to get intensity
        output = torch.abs(output_fft) ** 2

        # Normalize output
        batch_min = (
            output.view(output.size(0), -1)
            .min(dim=1, keepdim=True)[0]
            .unsqueeze(2)
            .unsqueeze(3)
        )
        batch_max = (
            output.view(output.size(0), -1)
            .max(dim=1, keepdim=True)[0]
            .unsqueeze(2)
            .unsqueeze(3)
        )
        output = (output - batch_min) / (batch_max - batch_min + 1e-8)

        # Verify output size
        assert (
            output.size() == x.size()
        ), f"Output size {output.size()} doesn't match input size {x.size()}"

        return output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Tuple of (decoded_object, probe_convolved_output)
                - decoded_object: Reconstructed object (B, 1, H, W)
                - probe_convolved_output: Diffraction pattern (B, 1, H, W)
        """
        # Encoder
        enc1_out = self.enc1(x)
        enc1_pooled = self.drop(self.pool1(enc1_out))

        enc2_out = self.enc2(enc1_pooled)
        enc2_pooled = self.drop(self.pool2(enc2_out))

        # Bottleneck
        bottleneck_out = self.bottleneck(enc2_pooled)

        # Decoder with skip connections
        up1_out = self.up1(bottleneck_out)
        dec1_out = self.dec1(torch.cat([up1_out, enc2_out], dim=1))

        up2_out = self.up2(dec1_out)
        dec2_out = self.dec2(torch.cat([up2_out, enc1_out], dim=1))

        # Final output
        decoded = self.sigmoid(self.final_layer(dec2_out))

        # Apply probe convolution
        probe_convolved_output = self.conv2d_probe(decoded)

        return decoded, probe_convolved_output
