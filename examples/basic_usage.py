#!/usr/bin/env python3
"""
Basic usage example for deconvolutionNN.

This example demonstrates how to use the package for training and inference.
"""


import numpy as np
import torch

# Import the main components
from deconvolutionNN import DeconvolutionEngine


def create_dummy_data(n_samples: int = 100, size: int = 128) -> np.ndarray:
    """
    Create dummy diffraction pattern data for demonstration.

    Args:
        n_samples: Number of samples to create
        size: Size of each diffraction pattern

    Returns:
        Array of dummy diffraction patterns
    """
    # Create random diffraction patterns with some structure
    data = np.random.rand(n_samples, size, size)

    # Add some structure to make it more realistic
    for i in range(n_samples):
        # Add central peak
        center = size // 2
        data[i, center - 5 : center + 5, center - 5 : center + 5] += 10

        # Add some random peaks
        for _ in range(3):
            x, y = np.random.randint(10, size - 10, 2)
            data[i, x - 2 : x + 2, y - 2 : y + 2] += 5

    return data


def create_dummy_probe(size: int = 128) -> np.ndarray:
    """
    Create a dummy probe kernel for demonstration.

    Args:
        size: Size of the probe kernel

    Returns:
        Complex probe kernel
    """
    # Create a simple Gaussian probe
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    center = size // 2

    # Gaussian function
    sigma = size // 8
    probe = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))

    # Add some phase variation
    phase = np.random.rand(size, size) * 2 * np.pi
    probe = probe * np.exp(1j * phase)

    return probe


def main() -> None:
    """Main example function."""
    print("🧠 deconvolutionNN - Basic Usage Example")
    print("=" * 50)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dummy data
    print("\n1. Creating dummy data...")
    data = create_dummy_data(n_samples=200, size=128)
    probe = create_dummy_probe(size=128)

    print(f"Data shape: {data.shape}")
    print(f"Probe shape: {probe.shape}")

    # Initialize the engine
    print("\n2. Initializing deconvolution engine...")
    engine = DeconvolutionEngine(device=device)

    # Load probe (in real usage, you'd load from file)
    print("\n3. Loading probe kernel...")
    engine.probe_kernel = probe

    # Create model
    print("\n4. Creating model...")
    model = engine.create_model()

    # Setup training
    print("\n5. Setting up training...")
    trainer = engine.setup_training(learning_rate=1e-4, weight_decay=1e-4)

    # Train model (with reduced epochs for demo)
    print("\n6. Training model...")
    print("Note: This is a demonstration with minimal training.")

    try:
        metrics = engine.train_model(
            data=data,
            batch_size=16,
            epochs=5,  # Very few epochs for demo
            train_split=0.8,
            val_split=0.1,
            loss_function="custom_loss",
            plot_samples=False,
            save_path="trained_models/demo_model.pth",
        )

        print("Training completed successfully!")
        print(f"Final training loss: {metrics['losses'][-1][0]:.6f}")
        print(f"Final validation loss: {metrics['val_losses'][-1][0]:.6f}")

    except Exception as e:
        print(f"Training failed (expected for demo): {e}")
        print("This is normal for the demo with dummy data.")

    # Demonstrate single deconvolution
    print("\n7. Demonstrating single deconvolution...")
    try:
        # Create a single test pattern
        test_pattern = data[0:1]  # Take first pattern

        # Deconvolve
        decoded, probe_convolved = engine.deconvolve(test_pattern)

        print(f"Input shape: {test_pattern.shape}")
        print(f"Decoded shape: {decoded.shape}")
        print(f"Probe convolved shape: {probe_convolved.shape}")

        print("Single deconvolution completed successfully!")

    except Exception as e:
        print(f"Deconvolution failed: {e}")

    # Plot training history if available
    if (
        hasattr(engine, "trainer")
        and engine.trainer
        and engine.trainer.metrics["losses"]
    ):
        print("\n8. Plotting training history...")
        try:
            engine.trainer.plot_training_history()
        except Exception as e:
            print(f"Could not plot training history: {e}")

    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nNext steps:")
    print("1. Use real diffraction pattern data")
    print("2. Load actual probe kernels from HDF5 files")
    print("3. Train for more epochs with proper validation")
    print("4. Use the web GUI: streamlit run src/deconvolutionNN/web/gui.py")


if __name__ == "__main__":
    main()
