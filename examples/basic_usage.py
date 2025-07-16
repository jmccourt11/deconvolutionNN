#!/usr/bin/env python3
"""
Basic usage example for deconvolutionNN with dummy data.

This example demonstrates how to use the package for training and inference
using generated dummy data and a dummy probe kernel.
"""

import logging

import numpy as np
import torch
from deconvolutionNN.core import (
    DeconvolutionEngine,
)
from deconvolutionNN.core.data_loader import create_center_mask, log10_custom
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_data(n_samples: int = 100, size: int = 256) -> np.ndarray:
    """
    Create dummy diffraction pattern data for demonstration.

    Args:
        n_samples: Number of diffraction patterns to generate
        size: Size of each diffraction pattern (size x size)

    Returns:
        Array of dummy diffraction patterns with shape (n_samples, size, size)
    """
    data = np.random.rand(n_samples, size, size)
    for i in range(n_samples):
        center = size // 2
        data[i, center - 5 : center + 5, center - 5 : center + 5] += 10
        for _ in range(3):
            x, y = np.random.randint(10, size - 10, 2)
            data[i, x - 2 : x + 2, y - 2 : y + 2] += 5
    return data


def create_dummy_probe(size: int = 256) -> np.ndarray:
    """
    Create a dummy probe kernel for demonstration.

    Args:
        size: Size of the probe kernel (size x size)

    Returns:
        Complex probe kernel array with shape (size, size)
    """
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    center = size // 2
    sigma = size // 8
    probe = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))
    phase = np.random.rand(size, size) * 2 * np.pi
    probe = probe * np.exp(1j * phase)
    return probe


def preprocess_dummy_data(
    data: np.ndarray,
    target_size: int = 256,
    center_radius: int = 40,
    min_intensity_threshold: float = 0.1,
) -> np.ndarray:
    """
    Preprocess dummy diffraction pattern data.

    Args:
        data: Input diffraction patterns
        target_size: Target size for resizing
        center_radius: Radius for central beam masking
        min_intensity_threshold: Minimum intensity threshold

    Returns:
        Preprocessed diffraction patterns
    """
    logger.info("Preprocessing dummy data...")
    logger.info(f"Original data shape: {data.shape}")

    # Apply log10 transformation
    logger.info("Applying log10 transformation...")
    try:
        amp_dps = log10_custom(data)
    except ImportError:
        amp_dps = np.log10(data + 1e-10)

    # Normalize data
    logger.info("Normalizing data...")
    amp_dps_norm = np.asarray(
        [
            (a - np.min(a)) / (np.max(a) - np.min(a))
            for a in tqdm(amp_dps, desc="Normalizing")
        ]
    )

    # Filter patterns using center mask
    logger.info("Filtering patterns...")
    mask = create_center_mask((target_size, target_size), center_radius)
    filtered_dps = []
    for dp in tqdm(amp_dps_norm, desc="Filtering"):
        total_intensity = np.sum(dp * mask)
        if total_intensity > min_intensity_threshold:
            filtered_dps.append(dp)
    processed = np.asarray(filtered_dps)

    logger.info(f"Preprocessed data shape: {processed.shape}")
    logger.info(f"Filtered out {len(amp_dps_norm) - len(filtered_dps)} patterns")
    return processed


def main() -> None:
    """Run the basic usage example with dummy data."""
    logger.info("🧠 deconvolutionNN - Basic Usage Example (Dummy Data)")
    logger.info("=" * 50)

    # Set device and random seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    logger.info(f"Using device: {device}")

    try:
        # Initialize the deconvolution engine
        logger.info("\n1. Initializing deconvolution engine...")
        engine = DeconvolutionEngine(device=device)

        # Create and load dummy data
        logger.info("\n2. Creating dummy data...")
        raw_data = create_dummy_data(n_samples=200, size=256)
        probe = create_dummy_probe(size=256)
        logger.info(f"Data shape: {raw_data.shape}")
        logger.info(f"Probe shape: {probe.shape}")

        # Load probe
        logger.info("\n3. Loading probe kernel...")
        engine.probe_kernel = probe
        logger.info("✓ Probe kernel loaded")

        # Create model
        logger.info("\n4. Creating model...")
        engine.create_model(model_type="conv_autoencoder_skip")
        logger.info("✓ Model created successfully")

        # Setup training
        logger.info("\n5. Setting up training...")
        engine.setup_training(
            learning_rate=1e-4, weight_decay=1e-4, training_mode="autoencoder"
        )
        logger.info("✓ Training setup completed")

        # Preprocess data
        logger.info("\n6. Preprocessing data...")
        processed_data = preprocess_dummy_data(
            data=raw_data,
            target_size=256,
            center_radius=40,
            min_intensity_threshold=0.1,
        )
        logger.info("✓ Data preprocessing completed")

        # Setup data splits and loaders
        logger.info("\n7. Setting up data splits and loaders...")
        engine.setup_data_splits_and_loaders(
            input_data=processed_data,
            target_data=processed_data,  # Same data for autoencoder mode
            batch_size=16,
            train_split=0.75,
            val_split=0.125,
            shuffle_train=True,
            random_state=42,
        )
        logger.info("✓ Data loaders created")

        # Train model
        logger.info("\n8. Training model...")
        metrics = engine.train_model(
            data=processed_data,  # Explicitly provide the processed data
            batch_size=16,
            epochs=5,  # Few epochs for demo
            train_split=0.75,
            val_split=0.125,
            loss_function="custom_loss",
            plot_samples=True,
            save_path="trained_models/demo_model.pth",
            preprocess_data=False,  # Data is already preprocessed
        )
        logger.info("✓ Training completed")
        logger.info(f"Final training loss: {metrics['losses'][-1][0]:.6f}")
        logger.info(f"Final validation loss: {metrics['val_losses'][-1][0]:.6f}")

        # Evaluate model
        logger.info("\n9. Evaluating model...")
        results, probe_convolved = engine.evaluate_model(
            data=processed_data,  # Explicitly provide the processed data
            batch_size=16,
            metric_names=["mse", "ssim", "psnr"],
        )
        logger.info("✓ Evaluation completed")

        # Plot results
        logger.info("\n10. Plotting results...")
        engine.plot_results(
            decoded_results=results,
            probe_convolved_results=probe_convolved,
            input_data=processed_data,  # Explicitly provide the processed data
            n_samples=3,
            mode="autoencoder",
        )
        logger.info("✓ Results plotted")

        # Plot training history
        logger.info("\n11. Plotting training history...")
        if engine.trainer and engine.trainer.metrics.get("losses"):
            engine.trainer.plot_training_history()
            logger.info("✓ Training history plotted")

        # Demonstrate single deconvolution
        logger.info("\n12. Testing single deconvolution...")
        test_pattern = processed_data[0:1]
        decoded, probe_convolved = engine.deconvolve(test_pattern)
        logger.info("Single deconvolution shapes:")
        logger.info(f"  Input: {test_pattern.shape}")
        logger.info(f"  Decoded: {decoded.shape}")
        logger.info(f"  Probe convolved: {probe_convolved.shape}")

        # Plot radial profiles
        logger.info("\n13. Plotting radial profiles...")
        engine.plot_radial_profiles(
            decoded_results=results,
            probe_convolved_results=probe_convolved,
            input_data=processed_data,  # Explicitly provide the processed data
            n_samples=3,
        )
        logger.info("✓ Radial profiles plotted")

        logger.info("\n" + "=" * 50)
        logger.info("Example completed successfully! 🎉")
        logger.info("\nNext steps:")
        logger.info("1. Use real diffraction pattern data")
        logger.info("2. Load actual probe kernels from HDF5 files")
        logger.info("3. Train for more epochs with proper validation")
        logger.info("4. Use the web GUI: streamlit run src/deconvolutionNN/web/gui.py")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
