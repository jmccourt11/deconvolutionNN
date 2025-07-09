"""Main deconvolution engine for neural network-based deconvolution."""

from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..models.autoencoder import AutoEncoder
from ..models.base import BaseModel
from ..models.conv_autoencoder import ConvAutoencoderSkip
from ..models.encoder1 import ReconModel
from ..models.encoder1_no_Unet import ReconModelNoUnet
from .data_loader import (
    create_data_loaders,
    load_convoluted_and_ideal_patterns,
    load_diffraction_patterns,
    load_probe_kernel,
    preprocess_diffraction_patterns,
    resize_probe,
)
from .train import DeconvolutionTrainer, azimuthal_average, evaluate_model


class DeconvolutionEngine:
    """
    Main engine for neural network-based deconvolution.

    This class provides a high-level interface for loading data, training models,
    and performing deconvolution on diffraction patterns.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        model_config: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the deconvolution engine.

        Args:
            device: Device to run computations on (CPU/GPU)
            model_config: Configuration dictionary for model parameters
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_config = model_config or {}
        self.model: Optional[ConvAutoencoderSkip] = None
        self.trainer: Optional[DeconvolutionTrainer] = None
        self.probe_kernel: Optional[np.ndarray] = None

        # Data storage
        self.conv_DPs: Optional[np.ndarray] = None
        self.ideal_DPs: Optional[np.ndarray] = None
        self.probe_DPs: Optional[np.ndarray] = None
        self.processed_data: Optional[np.ndarray] = None

        print(f"DeconvolutionEngine initialized on device: {self.device}")

    def load_convoluted_and_ideal_patterns(
        self, h5_file_path: Union[str, Path], max_dps: int = 10800
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load convoluted and ideal diffraction patterns from HDF5 file.

        This method loads three types of diffraction patterns:
        - convDP: Convoluted diffraction patterns (input data)
        - pinholeDP: Ideal diffraction patterns (target data)
        - probe_DPs: Dummy probe array for testing (placeholder)

        Args:
            h5_file_path: Path to the HDF5 file containing the diffraction patterns
            max_dps: Maximum number of diffraction patterns to load

        Returns:
            Tuple of (conv_DPs, ideal_DPs, probe_DPs)
                - conv_DPs: Convoluted diffraction patterns array
                - ideal_DPs: Ideal diffraction patterns array
                - probe_DPs: Dummy probe array for testing
        """
        print(f"Loading convoluted and ideal patterns from: {h5_file_path}")

        (
            self.conv_DPs,
            self.ideal_DPs,
            self.probe_DPs,
        ) = load_convoluted_and_ideal_patterns(
            h5_file_path=h5_file_path, max_dps=max_dps
        )

        print("Data loaded successfully:")
        print(f"  - Convoluted patterns: {self.conv_DPs.shape}")
        print(f"  - Ideal patterns: {self.ideal_DPs.shape}")
        print(f"  - Probe patterns: {self.probe_DPs.shape}")

        return self.conv_DPs, self.ideal_DPs, self.probe_DPs

    def load_probe(
        self, probe_path: Union[str, Path], target_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Load and optionally resize the probe kernel.

        Args:
            probe_path: Path to the probe kernel file
            target_size: Target size for resizing (if None, keeps original size)

        Returns:
            Loaded probe kernel
        """
        print(f"Loading probe kernel from: {probe_path}")
        self.probe_kernel = load_probe_kernel(probe_path)

        if target_size is not None:
            print(f"Resizing probe kernel to {target_size}x{target_size}")
            self.probe_kernel = resize_probe(self.probe_kernel, target_size)

        print(f"Probe kernel shape: {self.probe_kernel.shape}")
        return self.probe_kernel

    def load_data(
        self,
        data_dir: Union[str, Path],
        scan_numbers: list[int],
        center: np.ndarray,
        target_size: int = 128,
        center_radius: int = 40,
        min_intensity_threshold: float = 10000.0,
    ) -> np.ndarray:
        """
        Load and preprocess diffraction pattern data.

        Args:
            data_dir: Directory containing scan data
            scan_numbers: List of scan numbers to load
            center: Center coordinates for cropping
            target_size: Target size for resizing
            center_radius: Radius for central beam masking
            min_intensity_threshold: Minimum intensity threshold for filtering

        Returns:
            Preprocessed diffraction patterns
        """
        print(f"Loading diffraction patterns from scans: {scan_numbers}")

        # Load raw diffraction patterns
        dps = load_diffraction_patterns(data_dir, scan_numbers)
        print(f"Loaded {dps.shape[0]} diffraction patterns")

        # Preprocess the data
        self.processed_data = preprocess_diffraction_patterns(
            dps=dps,
            center=center,
            target_size=target_size,
            center_radius=center_radius,
            min_intensity_threshold=min_intensity_threshold,
        )

        print(f"Preprocessed data shape: {self.processed_data.shape}")
        return self.processed_data

    def create_model(self, model_type: str = "conv_autoencoder_skip") -> BaseModel:
        """
        Create the neural network model.

        Args:
            model_type: Which model architecture to use. Options are:
                - "conv_autoencoder_skip" (default)
                - "autoencoder"
                - "recon_model"
                - "recon_model_no_unet"

        Returns:
            Initialized model
        """
        if self.probe_kernel is None and model_type == "conv_autoencoder_skip":
            raise ValueError("Probe kernel must be loaded before creating model")

        if model_type == "conv_autoencoder_skip":
            print("Creating ConvAutoencoderSkip model")
            self.model = ConvAutoencoderSkip(self.probe_kernel)
        elif model_type == "autoencoder":
            print("Creating AutoEncoder model")
            self.model = AutoEncoder(self.model_config)
        elif model_type == "recon_model":
            print("Creating ReconModel (U-Net) model")
            self.model = ReconModel(self.model_config)
        elif model_type == "recon_model_no_unet":
            print("Creating ReconModelNoUnet (no U-Net) model")
            self.model = ReconModelNoUnet(self.model_config)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Move to device
        self.model = self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(
            f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)"
        )

        return self.model

    def setup_training(
        self, learning_rate: float = 1e-4, weight_decay: float = 1e-4
    ) -> DeconvolutionTrainer:
        """
        Setup the training infrastructure.

        Args:
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization

        Returns:
            Configured trainer
        """
        if self.model is None:
            raise ValueError("Model must be created before setting up training")

        print("Setting up training infrastructure")
        self.trainer = DeconvolutionTrainer(
            model=self.model,
            device=self.device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        return self.trainer

    def train_model(
        self,
        data: Optional[np.ndarray] = None,
        batch_size: int = 32,
        epochs: int = 100,
        train_split: float = 0.75,
        val_split: float = 0.125,
        loss_function: str = "custom_loss",
        plot_samples: bool = False,
        save_path: Optional[str] = None,
    ) -> dict[str, list]:
        """
        Train the deconvolution model.

        Args:
            data: Preprocessed diffraction patterns (if None, uses self.processed_data)
            batch_size: Batch size for training
            epochs: Number of training epochs
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            loss_function: Loss function to use ("custom_loss", "custom_loss2", "custom_loss3")
            plot_samples: Whether to plot sample predictions during training
            save_path: Path to save the best model

        Returns:
            Training metrics
        """
        if self.trainer is None:
            raise ValueError("Training must be set up before training")

        # Use stored data if none provided
        if data is None:
            if self.processed_data is None:
                raise ValueError(
                    "No data available. Load data first or provide data parameter."
                )
            data = self.processed_data

        print(f"Starting model training for {epochs} epochs")
        print(f"Data shape: {data.shape}, Batch size: {batch_size}")

        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            data=data,
            batch_size=batch_size,
            train_split=train_split,
            val_split=val_split,
        )

        # Setup scheduler
        iterations_per_epoch = len(train_loader)
        step_size = 6 * iterations_per_epoch
        max_lr = self.trainer.learning_rate * 10
        self.trainer.setup_scheduler(step_size, max_lr)

        # Select loss function
        from .losses import custom_loss, custom_loss2, custom_loss3

        loss_functions = {
            "custom_loss": custom_loss,
            "custom_loss2": custom_loss2,
            "custom_loss3": custom_loss3,
        }
        selected_loss = loss_functions.get(loss_function, custom_loss)

        # Train the model
        metrics = self.trainer.train(
            trainloader=train_loader,
            validloader=val_loader,
            epochs=epochs,
            loss_function=selected_loss,
            plot_samples=plot_samples,
            save_path=save_path,
        )

        print("Training completed")
        return metrics

    def evaluate_model(
        self,
        data: Optional[np.ndarray] = None,
        batch_size: int = 32,
        model_path: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the trained model.

        Args:
            data: Test data (if None, uses self.processed_data)
            batch_size: Batch size for evaluation
            model_path: Path to load model from (if None, uses current model)

        Returns:
            Tuple of (decoded_results, probe_convolved_results)
        """
        if model_path is not None:
            self.load_model(model_path)

        if self.model is None:
            raise ValueError("Model must be available for evaluation")

        # Use stored data if none provided
        if data is None:
            if self.processed_data is None:
                raise ValueError(
                    "No data available. Load data first or provide data parameter."
                )
            data = self.processed_data

        print("Evaluating model")

        # Create test data loader
        _, _, test_loader = create_data_loaders(
            data=data, batch_size=batch_size, train_split=0.8, val_split=0.1
        )

        # Evaluate
        results, results_pc = evaluate_model(
            model=self.model, testloader=test_loader, device=self.device
        )

        print(f"Evaluation completed. Results shape: {results.shape}")
        return results, results_pc

    def deconvolve(
        self, diffraction_pattern: np.ndarray, model_path: Optional[str] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Deconvolve a single diffraction pattern.

        Args:
            diffraction_pattern: Input diffraction pattern
            model_path: Path to load model from (if None, uses current model)

        Returns:
            Tuple of (decoded_object, probe_convolved_output)
        """
        if model_path is not None:
            self.load_model(model_path)

        if self.model is None:
            raise ValueError("Model must be available for deconvolution")

        # Prepare input
        if diffraction_pattern.ndim == 2:
            diffraction_pattern = diffraction_pattern[np.newaxis, np.newaxis, :, :]
        elif diffraction_pattern.ndim == 3:
            diffraction_pattern = diffraction_pattern[:, np.newaxis, :, :]

        # Convert to tensor
        input_tensor = torch.Tensor(diffraction_pattern).to(self.device)

        # Run inference
        self.model.eval()
        with torch.no_grad():
            decoded, probe_convolved = self.model(input_tensor)

        # Convert back to numpy
        decoded_np = decoded.cpu().numpy().squeeze()
        probe_convolved_np = probe_convolved.cpu().numpy().squeeze()

        return decoded_np, probe_convolved_np

    def save_model(self, path: str) -> None:
        """
        Save the current model.

        Args:
            path: Path to save the model
        """
        if self.trainer is None:
            raise ValueError("No trainer available to save model")

        self.trainer.save_model(path)
        print(f"Model saved to: {path}")

    def load_model(self, path: str) -> None:
        """
        Load a trained model.

        Args:
            path: Path to load the model from
        """
        if self.model is None:
            self.create_model()

        if self.trainer is None:
            self.setup_training()

        self.trainer.load_model(path)
        print(f"Model loaded from: {path}")

    def plot_results(
        self,
        decoded_results: np.ndarray,
        probe_convolved_results: np.ndarray,
        input_data: Optional[np.ndarray] = None,
        n_samples: int = 5,
    ) -> None:
        """
        Plot deconvolution results.

        Args:
            decoded_results: Decoded objects
            probe_convolved_results: Probe convolved outputs
            input_data: Original input data (if None, uses self.conv_DPs)
            n_samples: Number of samples to plot
        """
        # Use stored input data if none provided
        if input_data is None:
            if self.conv_DPs is None:
                raise ValueError(
                    "No input data available. Load data first or provide input_data parameter."
                )
            input_data = self.conv_DPs

        ntest = decoded_results.shape[0]
        n = min(n_samples, ntest)

        # Create figure
        fig, axes = plt.subplots(4, n, figsize=(15, 12))
        if n == 1:
            axes = axes.reshape(-1, 1)

        # Add labels
        fig.text(0.02, 0.8, "Input", fontsize=20)
        fig.text(0.02, 0.6, "Output", fontsize=20)
        fig.text(0.02, 0.4, "PC", fontsize=20)
        fig.text(0.02, 0.2, "Difference", fontsize=20)

        for i in range(n):
            j = int(round(np.random.rand() * ntest))

            # Input
            im = axes[0, i].imshow(
                input_data[j].reshape(input_data.shape[1], input_data.shape[2])
            )
            plt.colorbar(im, ax=axes[0, i], format="%.2f")
            axes[0, i].get_xaxis().set_visible(False)
            axes[0, i].get_yaxis().set_visible(False)

            # Decoded
            im = axes[1, i].imshow(
                decoded_results[j].reshape(
                    decoded_results.shape[1], decoded_results.shape[2]
                )
            )
            plt.colorbar(im, ax=axes[1, i], format="%.2f")
            axes[1, i].get_xaxis().set_visible(False)
            axes[1, i].get_yaxis().set_visible(False)

            # Probe convolved
            im = axes[2, i].imshow(
                probe_convolved_results[j].reshape(
                    probe_convolved_results.shape[1], probe_convolved_results.shape[2]
                )
            )
            plt.colorbar(im, ax=axes[2, i], format="%.2f")
            axes[2, i].get_xaxis().set_visible(False)
            axes[2, i].get_yaxis().set_visible(False)

            # Difference
            diff = input_data[j].reshape(
                input_data.shape[1], input_data.shape[2]
            ) - probe_convolved_results[j].reshape(
                probe_convolved_results.shape[1], probe_convolved_results.shape[2]
            )
            im = axes[3, i].imshow(diff)
            plt.colorbar(im, ax=axes[3, i], format="%.2f")
            axes[3, i].get_xaxis().set_visible(False)
            axes[3, i].get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_radial_profiles(
        self,
        decoded_results: np.ndarray,
        probe_convolved_results: np.ndarray,
        input_data: Optional[np.ndarray] = None,
        n_samples: int = 5,
    ) -> None:
        """
        Plot radial profiles of deconvolution results.

        Args:
            decoded_results: Decoded objects
            probe_convolved_results: Probe convolved outputs
            input_data: Original input data (if None, uses self.conv_DPs)
            n_samples: Number of samples to plot
        """
        # Use stored input data if none provided
        if input_data is None:
            if self.conv_DPs is None:
                raise ValueError(
                    "No input data available. Load data first or provide input_data parameter."
                )
            input_data = self.conv_DPs

        ntest = decoded_results.shape[0]
        n = min(n_samples, ntest)

        # Create figure
        fig, axes = plt.subplots(
            5, n, figsize=(20, 20), gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.2]}
        )
        if n == 1:
            axes = axes.reshape(-1, 1)

        # Add labels
        fig.text(0.02, 0.85, "Input", fontsize=20)
        fig.text(0.02, 0.67, "Output", fontsize=20)
        fig.text(0.02, 0.49, "PC", fontsize=20)
        fig.text(0.02, 0.31, "Difference", fontsize=20)
        fig.text(0.02, 0.13, "Radial Profiles", fontsize=20)

        for i in range(n):
            j = int(round(np.random.rand() * ntest))

            # Get images
            input_img = input_data[j].reshape(input_data.shape[1], input_data.shape[2])
            output_img = decoded_results[j].reshape(
                decoded_results.shape[1], decoded_results.shape[2]
            )
            pc_img = probe_convolved_results[j].reshape(
                probe_convolved_results.shape[1], probe_convolved_results.shape[2]
            )
            diff_img = input_img - pc_img

            # 2D plots
            im = axes[0, i].imshow(input_img)
            plt.colorbar(im, ax=axes[0, i], format="%.2f")
            axes[0, i].get_xaxis().set_visible(False)
            axes[0, i].get_yaxis().set_visible(False)

            im = axes[1, i].imshow(output_img)
            plt.colorbar(im, ax=axes[1, i], format="%.2f")
            axes[1, i].get_xaxis().set_visible(False)
            axes[1, i].get_yaxis().set_visible(False)

            im = axes[2, i].imshow(pc_img)
            plt.colorbar(im, ax=axes[2, i], format="%.2f")
            axes[2, i].get_xaxis().set_visible(False)
            axes[2, i].get_yaxis().set_visible(False)

            im = axes[3, i].imshow(diff_img)
            plt.colorbar(im, ax=axes[3, i], format="%.2f")
            axes[3, i].get_xaxis().set_visible(False)
            axes[3, i].get_yaxis().set_visible(False)

            # Calculate radial profiles
            center = np.array([input_img.shape[1] / 2, input_img.shape[0] / 2])
            r_input = azimuthal_average(input_img, center)
            r_output = azimuthal_average(output_img, center)
            r_pc = azimuthal_average(pc_img, center)
            r_diff = azimuthal_average(diff_img, center)

            # Plot radial profiles
            radii = np.arange(len(r_input))
            axes[4, i].plot(radii, r_output, "r--", label="Output")
            if i == 0:  # Only show legend for first plot
                axes[4, i].legend()
            axes[4, i].set_xlabel("Radius (pixels)")
            axes[4, i].set_ylabel("Intensity")
            axes[4, i].grid(True)

        plt.tight_layout()
        plt.show()
