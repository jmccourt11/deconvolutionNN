"""Evaluation utilities for deconvolution neural networks."""

from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.conv_autoencoder import ConvAutoencoderSkip
from .data_loader import create_data_loaders, create_paired_data_loaders


def evaluate_model(
    model: ConvAutoencoderSkip, 
    testloader: DataLoader, 
    device: torch.device,
    training_mode: str = "autoencoder"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the trained model on test data.

    Args:
        model: Trained model
        testloader: Test data loader
        device: Device to run evaluation on
        training_mode: "autoencoder" or "supervised"

    Returns:
        Tuple of (decoded_results, probe_convolved_results)
    """
    model.eval()
    results = []
    results_pc = []

    with torch.no_grad():
        for batch_data in tqdm(testloader, desc="Evaluating"):
            # Handle different data loader formats
            if training_mode == "supervised":
                # Paired data loader returns (input, target)
                tests, _ = batch_data
            else:
                # Single data loader returns (input,)
                tests = batch_data[0]
                
            tests = tests.to(device)
            model_output = model(tests)
            
            # Handle different model outputs
            if isinstance(model_output, tuple):
                decoded, probe_convolved = model_output
            else:
                # For models that return single tensor, use it as both decoded and probe_convolved
                decoded = model_output
                probe_convolved = model_output

            for j in range(tests.shape[0]):
                results.append(decoded[j].detach().cpu().numpy())
                results_pc.append(probe_convolved[j].detach().cpu().numpy())

    results = np.array(results).squeeze()
    results_pc = np.array(results_pc).squeeze()

    return results, results_pc


def evaluate_model_with_data(
    model: ConvAutoencoderSkip,
    input_data: np.ndarray,
    target_data: Optional[np.ndarray] = None,
    device: torch.device = None,
    batch_size: int = 32,
    training_mode: str = "autoencoder",
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model with provided data, handling both autoencoder and supervised modes.

    Args:
        model: Trained model
        input_data: Input data for evaluation
        target_data: Target data (required for supervised mode)
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        training_mode: "autoencoder" or "supervised"
        train_split: Fraction for training (not used in evaluation, but needed for data splitting)
        val_split: Fraction for validation (not used in evaluation, but needed for data splitting)

    Returns:
        Tuple of (decoded_results, probe_convolved_results)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create appropriate data loader
    if training_mode == "supervised":
        if target_data is None:
            raise ValueError("target_data is required for supervised mode evaluation")
        _, _, test_loader = create_paired_data_loaders(
            input_data=input_data,
            target_data=target_data,
            batch_size=batch_size,
            train_split=train_split,
            val_split=val_split,
        )
    else:
        _, _, test_loader = create_data_loaders(
            data=input_data,
            batch_size=batch_size,
            train_split=train_split,
            val_split=val_split,
        )

    return evaluate_model(model, test_loader, device, training_mode)


def evaluate_single_pattern(
    model: ConvAutoencoderSkip,
    diffraction_pattern: np.ndarray,
    device: torch.device = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a single diffraction pattern.

    Args:
        model: Trained model
        diffraction_pattern: Input diffraction pattern
        device: Device to run evaluation on

    Returns:
        Tuple of (decoded_object, probe_convolved_output)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare input
    if diffraction_pattern.ndim == 2:
        diffraction_pattern = diffraction_pattern[np.newaxis, np.newaxis, :, :]
    elif diffraction_pattern.ndim == 3:
        diffraction_pattern = diffraction_pattern[:, np.newaxis, :, :]

    # Convert to tensor
    input_tensor = torch.Tensor(diffraction_pattern).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        model_output = model(input_tensor)
        
        # Handle different model outputs
        if isinstance(model_output, tuple):
            decoded, probe_convolved = model_output
        else:
            # For models that return single tensor, use it as both decoded and probe_convolved
            decoded = model_output
            probe_convolved = model_output

    # Convert back to numpy
    decoded_np = decoded.cpu().numpy().squeeze()
    probe_convolved_np = probe_convolved.cpu().numpy().squeeze()

    return decoded_np, probe_convolved_np


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    metric_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Calculate evaluation metrics between predictions and targets.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        metric_names: List of metrics to calculate. Options: ['mse', 'mae', 'psnr', 'ssim']

    Returns:
        Dictionary of metric values
    """
    if metric_names is None:
        metric_names = ['mse', 'mae']
    
    metrics = {}
    
    # Ensure same shape
    if predictions.shape != targets.shape:
        raise ValueError(f"Predictions and targets must have same shape. Got {predictions.shape} and {targets.shape}")
    
    # Mean Squared Error
    if 'mse' in metric_names:
        metrics['mse'] = np.mean((predictions - targets) ** 2)
    
    # Mean Absolute Error
    if 'mae' in metric_names:
        metrics['mae'] = np.mean(np.abs(predictions - targets))
    
    # Peak Signal-to-Noise Ratio
    if 'psnr' in metric_names:
        mse = np.mean((predictions - targets) ** 2)
        if mse == 0:
            metrics['psnr'] = float('inf')
        else:
            max_val = np.max(targets)
            metrics['psnr'] = 20 * np.log10(max_val / np.sqrt(mse))
    
    # Structural Similarity Index
    if 'ssim' in metric_names:
        try:
            from skimage.metrics import structural_similarity as ssim
            # Calculate SSIM for each sample and average
            ssim_values = []
            for i in range(predictions.shape[0]):
                ssim_val = ssim(
                    predictions[i], 
                    targets[i], 
                    data_range=targets[i].max() - targets[i].min()
                )
                ssim_values.append(ssim_val)
            metrics['ssim'] = np.mean(ssim_values)
        except ImportError:
            print("Warning: skimage not available, skipping SSIM calculation")
    
    return metrics


def evaluate_model_full_dataset(
    model: ConvAutoencoderSkip,
    input_data: np.ndarray,
    target_data: Optional[np.ndarray] = None,
    device: torch.device = None,
    batch_size: int = 32,
    training_mode: str = "autoencoder",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on the full dataset (no train/val/test split).

    Args:
        model: Trained model
        input_data: Input data for evaluation
        target_data: Target data (required for supervised mode)
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        training_mode: "autoencoder" or "supervised"

    Returns:
        Tuple of (decoded_results, probe_convolved_results)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create data loaders for full dataset
    if training_mode == "supervised":
        if target_data is None:
            raise ValueError("target_data is required for supervised mode evaluation")
        
        # Create paired tensors for full dataset
        input_tensor = torch.Tensor(input_data.reshape(-1, 1, input_data.shape[1], input_data.shape[2]))
        target_tensor = torch.Tensor(target_data.reshape(-1, 1, target_data.shape[1], target_data.shape[2]))
        
        # Create dataset and loader for full data
        from torch.utils.data import TensorDataset, DataLoader
        full_dataset = TensorDataset(input_tensor, target_tensor)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    else:
        # Create single tensor for full dataset
        input_tensor = torch.Tensor(input_data.reshape(-1, 1, input_data.shape[1], input_data.shape[2]))
        
        # Create dataset and loader for full data
        from torch.utils.data import TensorDataset, DataLoader
        full_dataset = TensorDataset(input_tensor)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    return evaluate_model(model, full_loader, device, training_mode)


def evaluate_model_comprehensive(
    model: ConvAutoencoderSkip,
    input_data: np.ndarray,
    target_data: Optional[np.ndarray] = None,
    device: torch.device = None,
    batch_size: int = 32,
    training_mode: str = "autoencoder",
    metric_names: Optional[list[str]] = None,
    evaluate_full_dataset: bool = False,
    train_split: float = 0.75,
    val_split: float = 0.125,
) -> dict:
    """
    Comprehensive model evaluation with metrics.

    Args:
        model: Trained model
        input_data: Input data for evaluation
        target_data: Target data (required for supervised mode)
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        training_mode: "autoencoder" or "supervised"
        metric_names: List of metrics to calculate
        evaluate_full_dataset: If True, evaluate on full dataset; if False, use test split
        train_split: Fraction for training (must match training split)
        val_split: Fraction for validation (must match training split)

    Returns:
        Dictionary containing evaluation results and metrics
    """
    # Run evaluation
    if evaluate_full_dataset:
        decoded_results, probe_convolved_results = evaluate_model_full_dataset(
            model=model,
            input_data=input_data,
            target_data=target_data,
            device=device,
            batch_size=batch_size,
            training_mode=training_mode,
        )
    else:
        decoded_results, probe_convolved_results = evaluate_model_with_data(
            model=model,
            input_data=input_data,
            target_data=target_data,
            device=device,
            batch_size=batch_size,
            training_mode=training_mode,
            train_split=train_split,
            val_split=val_split,
        )
    
    # Calculate metrics if target data is available
    metrics = {}
    if target_data is not None:
        if training_mode == "supervised":
            # For supervised mode, compare decoded results with targets
            if evaluate_full_dataset:
                # Use full target data
                metrics = calculate_metrics(decoded_results, target_data, metric_names)
            else:
                # Use only test split of target data (same split as used in evaluation)
                n_total = len(target_data)
                n_train = int(n_total * train_split)
                n_val = int(n_total * val_split)
                test_targets = target_data[n_train + n_val:]
                metrics = calculate_metrics(decoded_results, test_targets, metric_names)
        else:
            # For autoencoder mode, compare probe_convolved results with inputs
            if evaluate_full_dataset:
                # Use full input data
                metrics = calculate_metrics(probe_convolved_results, input_data, metric_names)
            else:
                # Use only test split of input data (same split as used in evaluation)
                n_total = len(input_data)
                n_train = int(n_total * train_split)
                n_val = int(n_total * val_split)
                test_inputs = input_data[n_train + n_val:]
                metrics = calculate_metrics(probe_convolved_results, test_inputs, metric_names)
    
    return {
        'decoded_results': decoded_results,
        'probe_convolved_results': probe_convolved_results,
        'metrics': metrics,
        'training_mode': training_mode,
        'input_shape': input_data.shape,
        'output_shape': decoded_results.shape,
        'evaluated_on_full_dataset': evaluate_full_dataset,
        'test_split_info': {
            'train_split': train_split,
            'val_split': val_split,
            'test_split': 1.0 - train_split - val_split,
        } if not evaluate_full_dataset else None,
    }
