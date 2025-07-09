#!/usr/bin/env python3
"""
Basic usage example for deconvolutionNN with real data.

This example demonstrates how to use the package for training and inference
with actual diffraction pattern data and probe kernels.
"""

from pathlib import Path

import torch

# Import the main components
from deconvolutionNN.core.deconvolution import DeconvolutionEngine


def main() -> None:
    """Main example function."""
    print("🧠 deconvolutionNN - Real Data Usage Example")
    print("=" * 50)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create engine
    print("\n1. Initializing DeconvolutionEngine...")
    engine = DeconvolutionEngine()

    # Set path to diffraction patterns
    base_path = Path("/net/micdata/data2/12IDC/ptychosaxs")
    h5_file_path = base_path / "data/combined_data_TEMP.h5"

    # Load your diffraction pattern data
    print("\n2. Loading diffraction pattern data...")
    try:
        conv_DPs, ideal_DPs, probe_DPs = engine.load_convoluted_and_ideal_patterns(
            h5_file_path=h5_file_path,
            max_dps=500,  # Adjust based on your data size
        )
        print("✓ Diffraction patterns loaded successfully")
        
        # Plot example data samples
        print("\n2.5. Plotting example data samples...")
        engine.plot_data_samples(n_samples=3, use_preprocessed=False)  # Show raw data first
        engine.plot_data_samples(n_samples=3, use_preprocessed=True)   # Show preprocessed data
        
    except FileNotFoundError:
        print("✗ Could not find diffraction pattern file")
        print("  Please update the h5_file_path to point to your actual data file")
        return
    except Exception as e:
        print(f"✗ Error loading diffraction patterns: {e}")
        return

    # Load probe kernel
    print("\n3. Loading probe kernel...")
    try:
        engine.load_probe("path/to/your/probe_kernel.h5", target_size=256)
        print("✓ Probe kernel loaded successfully")
    except FileNotFoundError:
        print("✗ Could not find probe kernel file")
        print("  Please update the probe path to point to your actual probe file")
        return
    except Exception as e:
        print(f"✗ Error loading probe kernel: {e}")
        return

    # Create model
    print("\n4. Creating neural network model...")
    try:
        engine.create_model(model_type="recon_model")
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return

    # Setup training
    print("\n5. Setting up training infrastructure...")
    try:
        engine.setup_training(learning_rate=1e-4, weight_decay=1e-4, training_mode="supervised")
        print("✓ Training setup completed")
    except Exception as e:
        print(f"✗ Error setting up training: {e}")
        return

    # Train the model
    print("\n6. Training the model...")
    try:
        metrics = engine.train_model(
            epochs=8,  # Adjust based on your needs
            batch_size=32,  # Adjust based on your GPU memory
            train_split=0.75,
            val_split=0.125,
            loss_function="custom_loss",
            plot_samples=False,  # Set to False if you don't want plots during training
            save_path="trained_models/best_model.pth",
        )
        print("✓ Training completed successfully")
    except Exception as e:
        print(f"✗ Error during training: {e}")
        return

    # Evaluate the model
    print("\n7. Evaluating the model...")
    try:
        results, results_pc = engine.evaluate_model(
            batch_size=32,
            metric_names=['mse', 'mae', 'psnr']  # Add metrics calculation
        )
        print("✓ Evaluation completed successfully")
        print(f"  - Results shape: {results.shape}")
        print(f"  - Probe convolved shape: {results_pc.shape}")
        print(f"  - Training mode: {engine.training_mode}")
        print(f"  - Evaluated on test split (12.5% of data)")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        return

    # Plot results
    print("\n8. Plotting results...")
    try:
        engine.plot_results(
            decoded_results=results,
            probe_convolved_results=results_pc,
            input_data=conv_DPs,
            target_data=ideal_DPs,
            n_samples=3,
            # mode parameter is optional - will use trainer's mode automatically
        )
        print("✓ Results plotted successfully")
    except Exception as e:
        print(f"✗ Error plotting results: {e}")

    # Demonstrate single deconvolution
    print("\n9. Demonstrating single deconvolution...")
    try:
        # Use the first pattern from your loaded data
        test_pattern = conv_DPs[0:1]  # Take first pattern

        # Deconvolve
        decoded, probe_convolved = engine.deconvolve(test_pattern)

        print(f"Input shape: {test_pattern.shape}")
        print(f"Decoded shape: {decoded.shape}")
        print(f"Probe convolved shape: {probe_convolved.shape}")
        print("✓ Single deconvolution completed successfully!")

    except Exception as e:
        print(f"✗ Deconvolution failed: {e}")

    # Optional: Compare test split vs full dataset evaluation
    print("\n9.5. Optional: Full dataset evaluation for comparison...")
    try:
        from deconvolutionNN.core.eval import evaluate_model_comprehensive
        
        # Full dataset evaluation
        full_results = evaluate_model_comprehensive(
            model=engine.model,
            input_data=engine.processed_conv,
            target_data=engine.processed_ideal,
            device=engine.device,
            batch_size=32,
            training_mode=engine.training_mode,
            metric_names=['mse', 'mae', 'psnr'],
            evaluate_full_dataset=True,  # Use full dataset
        )
        
        print("Full dataset evaluation results:")
        for metric, value in full_results['metrics'].items():
            print(f"  - {metric}: {value:.6f}")
        print(f"  - Evaluated on {full_results['output_shape'][0]} samples (100% of data)")
        
    except Exception as e:
        print(f"✗ Full dataset evaluation failed: {e}")
        print("  (This is optional and not required for normal evaluation)")

    # Plot training history if available
    if (
        hasattr(engine, "trainer")
        and engine.trainer
        and hasattr(engine.trainer, "metrics")
        and engine.trainer.metrics.get("losses")
    ):
        print("\n10. Plotting training history...")
        try:
            engine.trainer.plot_training_history()
            print("✓ Training history plotted successfully")
        except Exception as e:
            print(f"✗ Could not plot training history: {e}")

    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nNext steps:")
    print("1. Adjust hyperparameters (learning rate, batch size, epochs)")
    print("2. Try different loss functions (custom_loss, custom_loss2, custom_loss3)")
    print("3. Experiment with different model architectures")
    print("4. Use the web GUI: streamlit run src/deconvolutionNN/web/gui.py")
    print("5. Save and load trained models for inference")


if __name__ == "__main__":
    main()
