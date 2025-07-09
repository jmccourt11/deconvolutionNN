"""Training utilities for deconvolution neural networks."""

import random
from collections.abc import Callable
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.conv_autoencoder import ConvAutoencoderSkip
from .losses import custom_loss


class DeconvolutionTrainer:
    """
    Trainer class for deconvolution neural networks.

    This class handles the training loop, validation, and model checkpointing
    for deconvolution models.
    """

    def __init__(
        self,
        model: ConvAutoencoderSkip,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        training_mode: str = "autoencoder",
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model: The neural network model to train
            device: Device to train on (CPU/GPU)
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            training_mode: "autoencoder" or "supervised"
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.training_mode = training_mode

        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Metrics tracking
        self.metrics = {
            "losses": [],
            "val_losses": [],
            "lrs": [],
            "best_val_loss": float("inf"),
        }

    def setup_scheduler(self, step_size: int, max_lr: float) -> None:
        """
        Setup the learning rate scheduler.

        Args:
            step_size: Number of steps per cycle
            max_lr: Maximum learning rate
        """
        self.scheduler = optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=self.learning_rate / 10,
            max_lr=max_lr,
            step_size_up=step_size,
            cycle_momentum=False,
            mode="triangular2",
        )

    def train_epoch(
        self,
        trainloader: DataLoader,
        loss_function: Callable = custom_loss,
        plot_samples: bool = False,
    ) -> float:
        """
        Train for one epoch.
        """
        self.model.train()
        tot_loss = 0.0

        for i, batch_data in enumerate(tqdm(trainloader, desc="Training")):
            # Handle different data loader formats
            if self.training_mode == "supervised":
                # Paired data loader returns (input, target)
                ft_images, target = batch_data
                ft_images = ft_images.to(self.device)
                target = target.to(self.device)
            else:
                # Single data loader returns (input,)
                ft_images = batch_data[0].to(self.device)
                target = ft_images  # For autoencoder mode
            
            model_output = self.model(ft_images)
            # Handle different model outputs
            if isinstance(model_output, tuple):
                decoded, probe_convolved = model_output
            else:
                decoded = model_output
                probe_convolved = model_output

            # Select prediction based on training_mode
            if self.training_mode == "supervised":
                prediction = decoded
            else:  # autoencoder mode
                prediction = probe_convolved

            self.optimizer.zero_grad()
            loss = loss_function(prediction, target, decoded)
            loss.backward()
            self.optimizer.step()
            tot_loss += loss.detach().item()

            if plot_samples and i == 0:
                self._plot_training_samples(ft_images, decoded, probe_convolved, mode=self.training_mode, target=target)
            if hasattr(self, "scheduler"):
                self.scheduler.step()
                self.metrics["lrs"].append(self.scheduler.get_last_lr())

        avg_loss = tot_loss / len(trainloader)
        self.metrics["losses"].append([avg_loss])
        return avg_loss

    def validate_epoch(
        self, validloader: DataLoader, loss_function: Callable = custom_loss
    ) -> float:
        """
        Validate for one epoch.
        """
        self.model.eval()
        tot_val_loss = 0.0

        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(validloader, desc="Validation")):
                # Handle different data loader formats
                if self.training_mode == "supervised":
                    # Paired data loader returns (input, target)
                    ft_images, target = batch_data
                    ft_images = ft_images.to(self.device)
                    target = target.to(self.device)
                else:
                    # Single data loader returns (input,)
                    ft_images = batch_data[0].to(self.device)
                    target = ft_images  # For autoencoder mode
                
                model_output = self.model(ft_images)
                if isinstance(model_output, tuple):
                    decoded, probe_convolved = model_output
                else:
                    decoded = model_output
                    probe_convolved = model_output

                # Select prediction based on training_mode
                if self.training_mode == "supervised":
                    prediction = decoded
                else:
                    prediction = probe_convolved

                val_loss = loss_function(prediction, target, decoded)
                tot_val_loss += val_loss.detach().item()

        avg_val_loss = tot_val_loss / len(validloader)
        self.metrics["val_losses"].append([avg_val_loss])
        if avg_val_loss < self.metrics["best_val_loss"]:
            self.metrics["best_val_loss"] = avg_val_loss
        return avg_val_loss

    def train(
        self,
        trainloader: DataLoader,
        validloader: DataLoader,
        epochs: int,
        loss_function: Callable = custom_loss,
        plot_samples: bool = False,
        save_path: Optional[str] = None,
    ) -> dict[str, list]:
        """
        Train the model for multiple epochs.

        Args:
            trainloader: Training data loader
            validloader: Validation data loader
            epochs: Number of epochs to train
            loss_function: Loss function to use
            plot_samples: Whether to plot sample predictions
            save_path: Path to save the best model

        Returns:
            Training metrics
        """
        print(f"Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(
                trainloader, loss_function=loss_function, plot_samples=plot_samples
            )

            # Validation
            val_loss = self.validate_epoch(validloader, loss_function=loss_function)

            # Print progress
            print(
                f"Epoch: {epoch:3d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
            )
            if hasattr(self, "scheduler"):
                print(f'Epoch: {epoch:3d} | LR: {self.metrics["lrs"][-1][0]:.6f}')

            # Save best model
            if save_path and val_loss == self.metrics["best_val_loss"]:
                self.save_model(save_path)
                print(f"Saved best model (val_loss: {val_loss:.5f})")

        return self.metrics

    def _plot_training_samples(
        self,
        ft_images: torch.Tensor,
        decoded: torch.Tensor,
        probe_convolved: torch.Tensor,
        mode: str = "autoencoder",
        target: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Plot sample predictions during training.
        Args:
            ft_images: Input images (convDP)
            decoded: Decoded objects (network output)
            probe_convolved: Probe convolved outputs
            mode: "autoencoder" or "supervised"
            target: Target data (for supervised mode, should be idealDP)
        """
        rand_idx = random.randint(0, ft_images.shape[0] - 1)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))
        # Always show input as convDP
        input_img = ft_images[rand_idx, 0].cpu().detach().numpy()
        im1 = ax1.imshow(input_img)
        ax1.set_title("Input (ConvDP)")
        plt.colorbar(im1, ax=ax1)
        if mode == "supervised" and target is not None:
            net_out = decoded[rand_idx, 0].cpu().detach().numpy()
            tgt = target[rand_idx, 0].cpu().detach().numpy()
            ax2.set_title("Network Output (Decoded)")
            ax3.set_title("Target (IdealDP)")
        else:
            net_out = probe_convolved[rand_idx, 0].cpu().detach().numpy()
            tgt = input_img
            ax2.set_title("Network Output (Probe Convolved)")
            ax3.set_title("Target (Input ConvDP)")
        im2 = ax2.imshow(net_out)
        plt.colorbar(im2, ax=ax2)
        im3 = ax3.imshow(tgt)
        plt.colorbar(im3, ax=ax3)
        diff = tgt - net_out
        im4 = ax4.imshow(diff)
        ax4.set_title("Difference (Target - Output)")
        plt.colorbar(im4, ax=ax4)
        plt.tight_layout()
        plt.show()

    def save_model(self, path: str) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": self.metrics,
                "model_config": {
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                },
            },
            path,
        )

    def load_model(self, path: str) -> None:
        """
        Load the model from disk.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.metrics = checkpoint["metrics"]

    def plot_training_history(self) -> None:
        """Plot training history."""
        if not self.metrics["losses"]:
            print("No training history to plot")
            return

        # Plot learning rate
        if self.metrics["lrs"]:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.plot(self.metrics["lrs"])
            plt.title("Learning Rate")
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.grid(True)

        # Plot losses
        plt.subplot(1, 2, 2)
        losses_arr = np.array(self.metrics["losses"])
        val_losses_arr = np.array(self.metrics["val_losses"])

        plt.plot(losses_arr[:, 0], "C3o-", label="Train Loss")
        plt.plot(val_losses_arr[:, 0], "C0o-", label="Val Loss")
        plt.title("Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def azimuthal_average(
    image: np.ndarray, center: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate azimuthal average of an image.

    Args:
        image: Input image
        center: Center coordinates (y, x). If None, uses image center

    Returns:
        Radial profile
    """
    # Get image dimensions
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    # Calculate radius for each pixel
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(np.int32)

    # Get sorted radii
    tbin = np.bincount(r.ravel(), weights=image.ravel())
    nr = np.bincount(r.ravel())

    # Avoid division by zero
    radialprofile = np.zeros_like(tbin)
    mask = nr > 0
    radialprofile[mask] = tbin[mask] / nr[mask]

    return radialprofile
