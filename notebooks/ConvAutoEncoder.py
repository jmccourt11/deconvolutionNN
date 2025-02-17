import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the Convolutional Autoencoder with Skip Connections
class ConvAutoencoderSkip(nn.Module):
    def __init__(self, probe_kernel):
        super(ConvAutoencoderSkip, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # Downsample

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # Downsample

        # Bottleneck (latent representation)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Decoder with Skip Connections
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # Concatenated input, so double channels
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),  # Skip connection doubles input channels
            nn.ReLU()
        )

        # Final reconstruction layer
        self.final_layer = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()  # Output values between 0 and 1

        # Probe Convolution (Fixed Kernel)
        self.register_buffer("probe_kernel", probe_kernel)

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc1_pooled = self.pool1(enc1_out)

        enc2_out = self.enc2(enc1_pooled)
        enc2_pooled = self.pool2(enc2_out)

        # Bottleneck
        bottleneck_out = self.bottleneck(enc2_pooled)

        # Decoder with Skip Connections
        up1_out = self.up1(bottleneck_out)
        dec1_out = self.dec1(torch.cat([up1_out, enc2_out], dim=1))  # Skip connection

        up2_out = self.up2(dec1_out)
        dec2_out = self.dec2(torch.cat([up2_out, enc1_out], dim=1))  # Skip connection

        # Final output
        decoded = self.sigmoid(self.final_layer(dec2_out))

        # Apply probe convolution
        probe_convolved_output = F.conv2d(decoded, self.probe_kernel, padding=1)

        return decoded, probe_convolved_output


# Create a fixed probe kernel (e.g., an edge detection filter)
probe_kernel = torch.tensor(
    [[[[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]]], 
    dtype=torch.float32
)  # Shape: (out_channels, in_channels, kernel_height, kernel_width)

# Initialize the model
model = ConvAutoencoderSkip(probe_kernel)

# Custom loss function (Mean Squared Error between probe-convoluted output and input)
def custom_loss(output, target):
    return F.mse_loss(output, target)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print model summary
print(model)