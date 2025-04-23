#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from typing import List, Dict, Tuple, Optional
import pandas as pd
import sys
import os
from matplotlib import colors
from skimage.metrics import structural_similarity as ssim
from scipy.signal import correlate2d, find_peaks
from scipy.ndimage import gaussian_filter
import random
import importlib
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../NN/ptychosaxsNN/')))
import utils.ptychosaxsNN_utils as ptNN_U
importlib.reload(ptNN_U)
# Add the models directory to the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../src/models/')))


def normalize_image(img: torch.Tensor) -> torch.Tensor:
    """
    Normalize image by subtracting background (minimum value) and scaling to [0, 1].
    
    Args:
        img (torch.Tensor): Input image
        
    Returns:
        torch.Tensor: Normalized image
    """
    # Convert to numpy for easier manipulation
    img_np = img.squeeze().cpu().numpy()
    
    # Subtract background (minimum value)
    img_np = img_np - np.min(img_np)
    
    # Scale to [0, 1]
    max_val = np.max(img_np)
    if max_val > 0:  # Avoid division by zero
        img_np = img_np / max_val
    
    return torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

def calculate_normalized_cross_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate 2D normalized cross-correlation between two images.
    
    Args:
        img1 (np.ndarray): First image
        img2 (np.ndarray): Second image
        
    Returns:
        float: Maximum normalized cross-correlation value
    """
    # Normalize images
    img1_norm = (img1 - np.mean(img1)) / (np.std(img1) * len(img1.ravel()))
    img2_norm = (img2 - np.mean(img2)) / np.std(img2)
    
    # Calculate cross-correlation
    corr = correlate2d(img1_norm, img2_norm, mode='same')
    
    # Return maximum correlation value
    return np.max(corr)

def find_peaks_and_fwhm(image: np.ndarray, threshold: float = 0.265, sigma: float = 0.714) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Find peaks and their FWHM in a 2D image, including peaks at edges.
    
    Args:
        image (np.ndarray): Input image
        threshold (float): Threshold for peak detection (relative to max)
        sigma (float): Sigma for Gaussian smoothing
        
    Returns:
        Tuple[List[Tuple[float, float]], List[float]]: Peak positions and FWHM values
    """
    # Smooth the image
    smoothed = gaussian_filter(image, sigma=sigma)
    
    # Find peaks
    peaks = []
    fwhm_values = []
    
    # Find local maxima
    max_val = np.max(smoothed)
    threshold_val = max_val * threshold
    
    height, width = smoothed.shape
    
    # Function to check if a point is a local maximum in its available neighborhood
    def is_local_max(i: int, j: int) -> bool:
        val = smoothed[i,j]
        
        # Define the neighborhood bounds, accounting for edges
        i_start = max(0, i-1)
        i_end = min(height, i+2)
        j_start = max(0, j-1)
        j_end = min(width, j+2)
        
        # Get the neighborhood
        neighborhood = smoothed[i_start:i_end, j_start:j_end]
        
        # For edge pixels, we only require them to be maximum in their partial neighborhood
        return val >= np.max(neighborhood)
    
    # Find peaks above threshold, including at edges
    for i in range(height):
        for j in range(width):
            if smoothed[i,j] > threshold_val and is_local_max(i, j):
                peaks.append((i, j))
                
                # Calculate FWHM in x and y directions
                x_profile = smoothed[i,:]
                y_profile = smoothed[:,j]
                center_val = smoothed[i,j]
                half_max = center_val / 2
                
                try:
                    # X direction FWHM
                    x_above = x_profile > half_max
                    
                    # Handle edge cases for X direction
                    if j == 0 or j == width-1:
                        # If peak is at edge, measure FWHM from the edge
                        x_fwhm = 2 * np.sum(x_above)  # Double to account for assumed symmetry
                    else:
                        x_transitions = np.where(x_above[:-1] != x_above[1:])[0]
                        if len(x_transitions) >= 2:
                            x_fwhm = x_transitions[-1] - x_transitions[0]
                        else:
                            x_fwhm = np.sum(x_above)
                    
                    # Y direction FWHM
                    y_above = y_profile > half_max
                    
                    # Handle edge cases for Y direction
                    if i == 0 or i == height-1:
                        # If peak is at edge, measure FWHM from the edge
                        y_fwhm = 2 * np.sum(y_above)  # Double to account for assumed symmetry
                    else:
                        y_transitions = np.where(y_above[:-1] != y_above[1:])[0]
                        if len(y_transitions) >= 2:
                            y_fwhm = y_transitions[-1] - y_transitions[0]
                        else:
                            y_fwhm = np.sum(y_above)
                    
                    # Use average of x and y FWHM
                    fwhm_values.append((x_fwhm + y_fwhm) / 2)
                    
                except Exception:
                    # Fallback method for FWHM calculation
                    x_fwhm = np.sum(x_profile > half_max)
                    y_fwhm = np.sum(y_profile > half_max)
                    fwhm_values.append((x_fwhm + y_fwhm) / 2)
    
    return peaks, fwhm_values

def calculate_peak_sensitivity_metrics(img1: np.ndarray, 
                                  img2: np.ndarray,
                                  sigma_range: List[float] = [0.5, 1.0, 1.5, 2.0],
                                  threshold_range: List[float] = [0.1, 0.2, 0.3, 0.4],
                                  distance_threshold: float = 5.0) -> Dict[str, float]:
    """
    Calculate comprehensive peak detection metrics across different peak finder parameters.
    
    Args:
        img1 (np.ndarray): First image (model output)
        img2 (np.ndarray): Second image (ground truth)
        sigma_range (List[float]): Range of sigma values for Gaussian smoothing
        threshold_range (List[float]): Range of threshold values for peak detection
        distance_threshold (float): Maximum distance for peaks to be considered matched
        
    Returns:
        Dict[str, float]: Dictionary of peak sensitivity metrics
    """
    metrics = {
        'optimal_sigma': 0.0,
        'optimal_threshold': 0.0,
        'max_f1_score': 0.0,
        'peak_position_stability': 0.0,
        'peak_count_stability': 0.0,
        'parameter_sensitivity': 0.0,
        'false_positive_rate': 0.0,
        'false_negative_rate': 0.0,
        'peak_intensity_correlation': 0.0,
        'peak_shape_consistency': 0.0
    }
    
    # Store results for each parameter combination
    results = []
    peak_positions_all = []
    peak_counts = []
    
    # Calculate ground truth peaks with middle parameters
    mid_sigma = np.median(sigma_range)
    mid_threshold = np.median(threshold_range)
    gt_peaks, gt_fwhm = find_peaks_and_fwhm(img2, threshold=mid_threshold, sigma=mid_sigma)
    
    # Test all parameter combinations
    for sigma in tqdm(sigma_range):
        for threshold in tqdm(threshold_range):
            # Find peaks with current parameters
            peaks, fwhm = find_peaks_and_fwhm(img1, threshold=threshold, sigma=sigma)
            peak_positions_all.extend(peaks)
            peak_counts.append(len(peaks))
            
            # Calculate matching metrics
            matched = 0
            false_positives = 0
            peak_intensities1 = []
            peak_intensities2 = []
            
            for peak in peaks:
                # Find closest ground truth peak
                min_dist = float('inf')
                closest_gt_peak = None
                
                for gt_peak in gt_peaks:
                    dist = np.sqrt((peak[0]-gt_peak[0])**2 + (peak[1]-gt_peak[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_gt_peak = gt_peak
                
                if min_dist <= distance_threshold:
                    matched += 1
                    peak_intensities1.append(img1[peak[0], peak[1]])
                    peak_intensities2.append(img2[closest_gt_peak[0], closest_gt_peak[1]])
                else:
                    false_positives += 1
            
            # Calculate F1 score
            precision = matched / len(peaks) if peaks else 0
            recall = matched / len(gt_peaks) if gt_peaks else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'sigma': sigma,
                'threshold': threshold,
                'f1_score': f1,
                'matched': matched,
                'false_positives': false_positives,
                'false_negatives': len(gt_peaks) - matched,
                'peak_count': len(peaks)
            })
            
            # Update best parameters if this combination gives better F1 score
            if f1 > metrics['max_f1_score']:
                metrics['max_f1_score'] = f1
                metrics['optimal_sigma'] = sigma
                metrics['optimal_threshold'] = threshold
    
    # Calculate stability metrics
    if peak_positions_all:
        # Peak position stability (standard deviation of peak positions across parameters)
        peak_positions_array = np.array(peak_positions_all)
        metrics['peak_position_stability'] = 1.0 / (np.std(peak_positions_array[:, 0]) + 
                                                  np.std(peak_positions_array[:, 1]) + 1e-6)
        
        # Peak count stability (coefficient of variation of peak counts)
        metrics['peak_count_stability'] = 1.0 / (np.std(peak_counts) / np.mean(peak_counts) + 1e-6)
    
    # Calculate parameter sensitivity
    f1_scores = [r['f1_score'] for r in results]
    metrics['parameter_sensitivity'] = np.std(f1_scores) / (np.mean(f1_scores) + 1e-6)
    
    # Calculate average false positive and negative rates
    metrics['false_positive_rate'] = np.mean([r['false_positives'] / r['peak_count'] 
                                            if r['peak_count'] > 0 else 0 for r in results])
    metrics['false_negative_rate'] = np.mean([r['false_negatives'] / len(gt_peaks) 
                                            if gt_peaks else 0 for r in results])
    
    return metrics

def calculate_metrics(img1: torch.Tensor, 
                     img2: torch.Tensor,
                     calculate_psnr: bool = True,
                     calculate_ssim: bool = True,
                     calculate_xcorr: bool = False,
                     calculate_peaks: bool = True,
                     calculate_peak_sensitivity: bool = False) -> Dict[str, float]:
    """
    Calculate selected metrics between two images.
    Images are normalized before metric calculation.
    
    Args:
        img1 (torch.Tensor): First image
        img2 (torch.Tensor): Second image
        calculate_psnr (bool): Whether to calculate PSNR
        calculate_ssim (bool): Whether to calculate SSIM
        calculate_xcorr (bool): Whether to calculate cross-correlation
        calculate_peaks (bool): Whether to calculate peak metrics
        calculate_peak_sensitivity (bool): Whether to calculate peak sensitivity metrics
        
    Returns:
        Dict[str, float]: Dictionary of calculated metrics
    """
    # Normalize both images
    img1_norm = normalize_image(img1)
    img2_norm = normalize_image(img2)
    
    # Convert to numpy arrays
    img1_np = img1_norm.squeeze().cpu().numpy()
    img2_np = img2_norm.squeeze().cpu().numpy()
    
    metrics = {}
    
    # Calculate selected metrics
    if calculate_psnr:
        mse = np.mean((img1_np - img2_np) ** 2)
        if mse == 0:
            metrics['psnr'] = float('inf')
        else:
            max_pixel = 1.0
            metrics['psnr'] = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    if calculate_ssim:
        metrics['ssim'] = ssim(img1_np, img2_np, data_range=1.0)
    
    if calculate_xcorr:
        metrics['xcorr'] = calculate_normalized_cross_correlation(img1_np, img2_np)
    
    if calculate_peaks:
        # Find peaks in both images
        peaks1, fwhm1 = find_peaks_and_fwhm(img1_np)
        peaks2, fwhm2 = find_peaks_and_fwhm(img2_np)
        
        # Calculate peak position differences
        if peaks1 and peaks2:
            # Find closest matching peaks
            peak_diffs = []
            fwhm_diffs = []
            
            # For each peak in image 1, find closest peak in image 2
            for p1, f1 in zip(peaks1, fwhm1):
                min_dist = float('inf')
                closest_f2 = None
                for p2, f2 in zip(peaks2, fwhm2):
                    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_f2 = f2
                peak_diffs.append(min_dist)
                if closest_f2 is not None:
                    fwhm_diffs.append(abs(f1 - closest_f2))
            
            metrics['avg_peak_dist'] = np.mean(peak_diffs)
            metrics['max_peak_dist'] = np.max(peak_diffs)
            metrics['num_peaks1'] = len(peaks1)
            metrics['num_peaks2'] = len(peaks2)
            if fwhm_diffs:  # Only calculate if we found matching peaks
                metrics['fwhm_diff'] = np.mean(fwhm_diffs)
    
    if calculate_peak_sensitivity:
        # Calculate peak sensitivity metrics
        sensitivity_metrics = calculate_peak_sensitivity_metrics(
            img1_np, 
            img2_np,
            sigma_range=[0.5, 1.0, 1.5, 2.0],
            threshold_range=[0.1, 0.2, 0.3, 0.4]
        )
        
        # Add sensitivity metrics to the output
        metrics.update({
            'optimal_sigma': sensitivity_metrics['optimal_sigma'],
            'optimal_threshold': sensitivity_metrics['optimal_threshold'],
            'max_f1_score': sensitivity_metrics['max_f1_score'],
            'peak_position_stability': sensitivity_metrics['peak_position_stability'],
            'peak_count_stability': sensitivity_metrics['peak_count_stability'],
            'parameter_sensitivity': sensitivity_metrics['parameter_sensitivity'],
            'false_positive_rate': sensitivity_metrics['false_positive_rate'],
            'false_negative_rate': sensitivity_metrics['false_negative_rate']
        })
    
    return metrics

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1 (torch.Tensor): First image
        img2 (torch.Tensor): Second image
        
    Returns:
        float: PSNR value in dB
    """
    # Convert to numpy arrays
    img1 = img1.squeeze().cpu().numpy()
    img2 = img2.squeeze().cpu().numpy()
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    max_pixel = 1.0  # Assuming normalized images
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def load_model(model_path: str, use_unet: bool = True) -> torch.nn.Module:
    """
    Load a trained model from the specified path.
    
    Args:
        model_path (str): Path to the saved model
        use_unet (bool): Whether to use Unet architecture
        
    Returns:
        torch.nn.Module: Loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize appropriate model class
    if use_unet:
        from encoder1 import recon_model
        model = recon_model()
    else:
        from encoder1_no_Unet import recon_model
        model = recon_model()
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    return model

def get_model_paths_from_config(model_configs: Dict, base_path: str = "") -> List[Dict]:
    """
    Generate model paths and metadata from model_configs dictionary.
    
    Args:
        model_configs (Dict): Dictionary containing model configurations
        base_path (str): Base path where models are stored
        
    Returns:
        List[Dict]: List of dictionaries containing model paths and metadata
    """
    model_info = []
    
    for model_name, model_template in model_configs['models'].items():
        # Parse model name to get architecture and loss type
        parts = model_name.split('_')
        loss_type = parts[0]  # L2 or pearson
        use_unet = 'no_Unet' not in model_name
        
        for iteration in model_configs['iterations']:
            model_path = Path(base_path) / model_template.format(iteration)
            if model_path.exists():
                model_info.append({
                    'path': str(model_path),
                    'loss_type': loss_type,
                    'use_unet': use_unet,
                    'iterations': iteration
                })
    
    return model_info

def create_comparison_grid_from_config(model_configs: Dict,
                                     input_data: torch.Tensor,
                                     ideal_data: torch.Tensor,
                                     base_path: str = "",
                                     figsize: Tuple[int, int] = (20, 15),
                                     calculate_psnr: bool = True,
                                     calculate_ssim: bool = True,
                                     calculate_xcorr: bool = False,
                                     calculate_peaks: bool = True,
                                     peak_sigma: float = 1.0) -> plt.Figure:
    """
    Create a grid plot comparing outputs from different models using model_configs.
    Shows input and ideal images above the model comparison grid.
    Includes selected metrics for each model output compared to the ideal image.
    
    Args:
        model_configs (Dict): Dictionary containing model configurations
        input_data (torch.Tensor): Input data to run through the models
        ideal_data (torch.Tensor): Ideal/ground truth data for comparison
        base_path (str): Base path where models are stored
        figsize (Tuple[int, int]): Figure size
        calculate_psnr (bool): Whether to calculate PSNR
        calculate_ssim (bool): Whether to calculate SSIM
        calculate_xcorr (bool): Whether to calculate cross-correlation
        calculate_peaks (bool): Whether to calculate peak metrics
        peak_sigma (float): Sigma for Gaussian smoothing in peak detection
        
    Returns:
        plt.Figure: The created figure
    """
    # Get model paths and metadata
    model_info = get_model_paths_from_config(model_configs, base_path)
    
    if not model_info:
        raise ValueError("No valid model paths found in the configuration")
    
    # Create DataFrame for easier organization
    df = pd.DataFrame(model_info)
    
    # Determine grid dimensions
    nrows = len(model_configs['iterations'])
    ncols = len(model_configs['models'])
    
    # Adjust figure size based on number of columns
    width_per_col = 5  # Width per column in inches
    height_per_row = 4  # Height per row in inches
    figsize = (width_per_col * ncols, height_per_row * (nrows + 1))
    
    # Create figure with extra row for input and ideal images
    fig = plt.figure(figsize=figsize)
    
    # Create a grid for the entire figure with adjusted spacing and height ratios
    gs = fig.add_gridspec(nrows + 1, ncols, 
                         height_ratios=[0.7] + [1]*nrows,  # Make top row slightly smaller
                         hspace=0.3, wspace=0.3)
    
    # Plot input and ideal images in the first two columns of the top row
    ax_input = fig.add_subplot(gs[0, 0])
    ax_ideal = fig.add_subplot(gs[0, 1])
    
    # Plot input image
    im_input = ax_input.imshow(input_data.squeeze().cpu().numpy(), cmap='jet')
    ax_input.set_title('Input Image')
    plt.colorbar(im_input, ax=ax_input, fraction=0.046, pad=0.04)
    
    # Plot ideal image
    im_ideal = ax_ideal.imshow(ideal_data.squeeze().cpu().numpy(), cmap='jet')
    ax_ideal.set_title('Ideal Image')
    plt.colorbar(im_ideal, ax=ax_ideal, fraction=0.046, pad=0.04)
    
    # Find peaks in ideal image if peak calculation is enabled
    ideal_peaks = None
    if calculate_peaks:
        ideal_peaks, ideal_fwhm = find_peaks_and_fwhm(ideal_data.squeeze().cpu().numpy(), sigma=peak_sigma)
        # Plot peaks on ideal image
        for peak in ideal_peaks:
            ax_ideal.plot(peak[1], peak[0], 'g+', markersize=8, markeredgewidth=2)
    
    # Create axes for model outputs
    axes = np.zeros((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[i+1, j])
    
    # Sort by iterations and model name for consistent plotting
    df = df.sort_values(['iterations', 'loss_type'])
    
    # Plot each model's output
    for idx, (_, row) in enumerate(df.iterrows()):
        row_idx = model_configs['iterations'].index(row['iterations'])
        col_idx = list(model_configs['models'].keys()).index(
            f"{row['loss_type']}_{'Unet' if row['use_unet'] else 'no_Unet'}")
        
        # Load and run model
        model = load_model(row['path'], row['use_unet'])
        with torch.no_grad():
            output = model(input_data)
        
        # Calculate selected metrics
        metrics = calculate_metrics(output, ideal_data,
                                  calculate_psnr=calculate_psnr,
                                  calculate_ssim=calculate_ssim,
                                  calculate_xcorr=calculate_xcorr,
                                  calculate_peaks=calculate_peaks)
        
        # Plot
        ax = axes[row_idx, col_idx]
        im = ax.imshow(output.squeeze().cpu().numpy(), cmap='jet')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot peaks if peak calculation is enabled
        matched_peaks = 0
        if calculate_peaks and ideal_peaks:
            # Plot ideal peaks in green
            for peak in ideal_peaks:
                ax.plot(peak[1], peak[0], 'g+', markersize=8, markeredgewidth=2)
            
            # Find peaks in model output
            output_peaks, _ = find_peaks_and_fwhm(output.squeeze().cpu().numpy(), sigma=peak_sigma)
            
            # Only plot output peaks that are close to ideal peaks
            for ideal_peak in ideal_peaks:
                min_dist = float('inf')
                closest_output_peak = None
                
                for output_peak in output_peaks:
                    dist = np.sqrt((ideal_peak[0]-output_peak[0])**2 + (ideal_peak[1]-output_peak[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_output_peak = output_peak
                
                # If we found a close peak (within 5 pixels), plot it
                if closest_output_peak is not None and min_dist < 5:
                    ax.plot(closest_output_peak[1], closest_output_peak[0], 'rx', markersize=8, markeredgewidth=2)
                    matched_peaks += 1
        
        # Build metrics text
        metrics_text = []
        if calculate_psnr:
            metrics_text.append(f'PSNR: {metrics["psnr"]:.2f} dB')
        if calculate_ssim:
            metrics_text.append(f'SSIM: {metrics["ssim"]:.4f}')
        if calculate_xcorr:
            metrics_text.append(f'XCORR: {metrics["xcorr"]:.4f}')
        if calculate_peaks and 'avg_peak_dist' in metrics:
            metrics_text.append(f'Peak Dist: {metrics["avg_peak_dist"]:.2f}')
            metrics_text.append(f'FWHM Diff: {metrics["fwhm_diff"]:.2f}')
            if ideal_peaks:
                metrics_text.append(f'Matched: {matched_peaks}/{len(ideal_peaks)}')
                #metrics_text.append(f'Sigma: {peak_sigma:.1f}')
        
        # Add metrics text in the corner
        ax.text(0.02, 0.98, '\n'.join(metrics_text),
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add row labels for iterations
        if col_idx == 0:
            ax.set_ylabel(f"{row['iterations']} iterations")
        
        # Add column labels for model types
        if row_idx == 0:
            model_type = "Unet" if row['use_unet'] else "No Unet"
            ax.set_title(f"{row['loss_type']}\n{model_type}")
    
    # Add legend for peak markers in a better position
    if calculate_peaks:
        legend_elements = [
            plt.Line2D([0], [0], marker='+', color='g', label='Ideal Peaks', markersize=8, linestyle='None'),
            plt.Line2D([0], [0], marker='x', color='r', label='Matched Model Peaks', markersize=8, linestyle='None')
        ]
        # Place legend in the empty space in the top row
        if len(model_configs['models']) > 2:
            ax_legend = fig.add_subplot(gs[0, 2:])
            ax_legend.axis('off')
            ax_legend.legend(handles=legend_elements, loc='center', frameon=False)
    
    plt.tight_layout()
    return fig

def save_comparison_grid(fig: plt.Figure, save_path: str):
    """
    Save the comparison grid plot.
    
    Args:
        fig (plt.Figure): Figure to save
        save_path (str): Path where to save the figure
    """
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def calculate_cumulative_stats(model_configs: Dict,
                           indices_list: List[Tuple[int, int, int]],
                           base_path: str = "",
                           mask_path: str = '/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy',
                           data_path: str = '/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls400_gs1024_lsp6_r3.0_typeSC',
                           calculate_psnr: bool = True,
                           calculate_ssim: bool = True,
                           calculate_xcorr: bool = False,
                           calculate_peaks: bool = True,
                           calculate_peak_sensitivity: bool = False,
                           peak_sigma: float = 1.0,
                           central_only: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Calculate cumulative statistics across multiple input patterns.
    
    Args:
        model_configs (Dict): Dictionary containing model configurations
        indices_list (List[Tuple[int, int, int]]): List of (hr, kr, lr) indices
        base_path (str): Base path where models are stored
        mask_path (str): Path to the mask file
        data_path (str): Path to the data directory
        calculate_psnr (bool): Whether to calculate PSNR
        calculate_ssim (bool): Whether to calculate SSIM
        calculate_xcorr (bool): Whether to calculate cross-correlation
        calculate_peaks (bool): Whether to calculate peak metrics
        calculate_peak_sensitivity (bool): Whether to calculate peak sensitivity metrics
        peak_sigma (float): Sigma for Gaussian smoothing in peak detection
        central_only (bool): If True, only process central pattern (num=5), 
                           if False, process all patterns (num=1-9)
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary of cumulative statistics per model
    """
    # Load mask
    mask = np.load(mask_path)
    
    # Initialize statistics dictionary
    stats = {}
    
    # Get model paths
    model_info = get_model_paths_from_config(model_configs, base_path)
    if not model_info:
        raise ValueError("No valid model paths found in the configuration")
    
    # Create DataFrame for easier organization
    df = pd.DataFrame(model_info)
    df = df.sort_values(['iterations', 'loss_type'])
    
    # Initialize metrics for each model
    for _, row in df.iterrows():
        model_key = f"{row['loss_type']}_{'Unet' if row['use_unet'] else 'no_Unet'}_{row['iterations']}"
        stats[model_key] = {
            'psnr_sum': 0.0,
            'ssim_sum': 0.0,
            'xcorr_sum': 0.0,
            'peak_dist_sum': 0.0,
            'fwhm_diff_sum': 0.0,
            'total_peaks_ideal': 0,
            'total_peaks_matched': 0,
            'pattern_count': 0,
            # Initialize peak sensitivity metrics if enabled
            'optimal_sigma_sum': 0.0,
            'optimal_threshold_sum': 0.0,
            'max_f1_score_sum': 0.0,
            'peak_position_stability_sum': 0.0,
            'peak_count_stability_sum': 0.0,
            'parameter_sensitivity_sum': 0.0,
            'false_positive_rate_sum': 0.0,
            'false_negative_rate_sum': 0.0
        }
    
    # Process each pattern
    for hr, kr, lr in indices_list:
        # Define which pattern numbers to process
        pattern_nums = [5] if central_only else range(1, 10)
        
        for num in pattern_nums:
            # Load and preprocess data
            pattern_file = f'output_hanning_conv_{hr}_{kr}_{lr}_0000{num}.npz'
            try:
                data = np.load(f'{data_path}/{pattern_file}')
            except FileNotFoundError:
                print(f"Warning: File not found: {pattern_file}")
                continue
            
            dp_pp, _, _ = ptNN_U.preprocess_ZCB_9(data['convDP'], mask)
            dp_pp_IDEAL, _, _ = ptNN_U.preprocess_ZCB_9(data['pinholeDP_extra_conv'], mask=np.ones(dp_pp[0][0].shape))
            
            # Convert to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dp_pp = dp_pp.to(device=device, dtype=torch.float)
            dp_pp_IDEAL = dp_pp_IDEAL.to(device=device, dtype=torch.float)
            
            # Process each model
            for _, row in df.iterrows():
                model_key = f"{row['loss_type']}_{'Unet' if row['use_unet'] else 'no_Unet'}_{row['iterations']}"
                
                # Load and run model
                model = load_model(row['path'], row['use_unet'])
                with torch.no_grad():
                    output = model(dp_pp)
                
                # Calculate metrics
                metrics = calculate_metrics(output, dp_pp_IDEAL,
                                         calculate_psnr=calculate_psnr,
                                         calculate_ssim=calculate_ssim,
                                         calculate_xcorr=calculate_xcorr,
                                         calculate_peaks=calculate_peaks,
                                         calculate_peak_sensitivity=calculate_peak_sensitivity)
                
                # Update statistics
                if calculate_psnr and 'psnr' in metrics:
                    stats[model_key]['psnr_sum'] += metrics['psnr']
                if calculate_ssim and 'ssim' in metrics:
                    stats[model_key]['ssim_sum'] += metrics['ssim']
                if calculate_xcorr and 'xcorr' in metrics:
                    stats[model_key]['xcorr_sum'] += metrics['xcorr']
                if calculate_peaks:
                    if 'avg_peak_dist' in metrics:
                        stats[model_key]['peak_dist_sum'] += metrics['avg_peak_dist']
                    if 'fwhm_diff' in metrics:
                        stats[model_key]['fwhm_diff_sum'] += metrics['fwhm_diff']
                    if 'num_peaks2' in metrics:  # ideal peaks
                        stats[model_key]['total_peaks_ideal'] += metrics['num_peaks2']
                        # Count matched peaks (those within distance threshold)
                        ideal_peaks, _ = find_peaks_and_fwhm(dp_pp_IDEAL.squeeze().cpu().numpy(), sigma=peak_sigma)
                        output_peaks, _ = find_peaks_and_fwhm(output.squeeze().cpu().numpy(), sigma=peak_sigma)
                        matched = 0
                        for ideal_peak in ideal_peaks:
                            min_dist = float('inf')
                            for output_peak in output_peaks:
                                dist = np.sqrt((ideal_peak[0]-output_peak[0])**2 + (ideal_peak[1]-output_peak[1])**2)
                                min_dist = min(min_dist, dist)
                            if min_dist < 5:  # Same threshold as in visualization
                                matched += 1
                        stats[model_key]['total_peaks_matched'] += matched
                
                if calculate_peak_sensitivity:
                    # Add peak sensitivity metrics
                    stats[model_key]['optimal_sigma_sum'] += metrics['optimal_sigma']
                    stats[model_key]['optimal_threshold_sum'] += metrics['optimal_threshold']
                    stats[model_key]['max_f1_score_sum'] += metrics['max_f1_score']
                    stats[model_key]['peak_position_stability_sum'] += metrics['peak_position_stability']
                    stats[model_key]['peak_count_stability_sum'] += metrics['peak_count_stability']
                    stats[model_key]['parameter_sensitivity_sum'] += metrics['parameter_sensitivity']
                    stats[model_key]['false_positive_rate_sum'] += metrics['false_positive_rate']
                    stats[model_key]['false_negative_rate_sum'] += metrics['false_negative_rate']
                
                stats[model_key]['pattern_count'] += 1
    
    # Calculate averages and create final statistics
    final_stats = {}
    for model_key, model_stats in stats.items():
        count = model_stats['pattern_count']
        if count > 0:
            final_stats[model_key] = {
                'avg_psnr': model_stats['psnr_sum'] / count,
                'avg_ssim': model_stats['ssim_sum'] / count,
                'avg_xcorr': model_stats['xcorr_sum'] / count,
                'avg_peak_dist': model_stats['peak_dist_sum'] / count,
                'avg_fwhm_diff': model_stats['fwhm_diff_sum'] / count,
                'peak_detection_rate': model_stats['total_peaks_matched'] / model_stats['total_peaks_ideal'] if model_stats['total_peaks_ideal'] > 0 else 0,
                'total_peaks_matched': model_stats['total_peaks_matched'],
                'total_peaks_ideal': model_stats['total_peaks_ideal'],
                'patterns_processed': count
            }
            
            if calculate_peak_sensitivity:
                final_stats[model_key].update({
                    'avg_optimal_sigma': model_stats['optimal_sigma_sum'] / count,
                    'avg_optimal_threshold': model_stats['optimal_threshold_sum'] / count,
                    'avg_max_f1_score': model_stats['max_f1_score_sum'] / count,
                    'avg_peak_position_stability': model_stats['peak_position_stability_sum'] / count,
                    'avg_peak_count_stability': model_stats['peak_count_stability_sum'] / count,
                    'avg_parameter_sensitivity': model_stats['parameter_sensitivity_sum'] / count,
                    'avg_false_positive_rate': model_stats['false_positive_rate_sum'] / count,
                    'avg_false_negative_rate': model_stats['false_negative_rate_sum'] / count
                })
    
    return final_stats

def group_stats_by_model_type(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Group statistics by model type (L1/L2/pearson, Unet/no_Unet), combining iterations.
    
    Args:
        stats (Dict[str, Dict[str, float]]): Dictionary of statistics per model
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary of grouped statistics per model type
    """
    grouped_stats = {}
    
    for model_key, metrics in stats.items():
        # Extract model type (e.g., 'L1_Unet', 'L2_no_Unet', etc.)
        model_type = '_'.join(model_key.split('_')[:-1])  # Remove iteration number
        
        if model_type not in grouped_stats:
            grouped_stats[model_type] = {
                'avg_psnr': [],
                'avg_ssim': [],
                'avg_xcorr': [],
                'avg_peak_dist': [],
                'avg_fwhm_diff': [],
                'peak_detection_rate': [],
                'total_peaks_matched': 0,
                'total_peaks_ideal': 0,
                'total_patterns': 0
            }
        
        # Append individual metrics to lists for later averaging
        for metric in ['avg_psnr', 'avg_ssim', 'avg_xcorr', 'avg_peak_dist', 'avg_fwhm_diff', 'peak_detection_rate']:
            if metric in metrics:
                grouped_stats[model_type][metric].append(metrics[metric])
        
        # Sum up total peaks and patterns
        grouped_stats[model_type]['total_peaks_matched'] += metrics['total_peaks_matched']
        grouped_stats[model_type]['total_peaks_ideal'] += metrics['total_peaks_ideal']
        grouped_stats[model_type]['total_patterns'] += metrics['patterns_processed']
    
    # Calculate averages for each model type
    final_grouped_stats = {}
    for model_type, metrics in grouped_stats.items():
        final_grouped_stats[model_type] = {
            'avg_psnr': np.mean(metrics['avg_psnr']) if metrics['avg_psnr'] else 0,
            'avg_ssim': np.mean(metrics['avg_ssim']) if metrics['avg_ssim'] else 0,
            'avg_xcorr': np.mean(metrics['avg_xcorr']) if metrics['avg_xcorr'] else 0,
            'avg_peak_dist': np.mean(metrics['avg_peak_dist']) if metrics['avg_peak_dist'] else 0,
            'avg_fwhm_diff': np.mean(metrics['avg_fwhm_diff']) if metrics['avg_fwhm_diff'] else 0,
            'peak_detection_rate': metrics['total_peaks_matched'] / metrics['total_peaks_ideal'] if metrics['total_peaks_ideal'] > 0 else 0,
            'total_peaks_matched': metrics['total_peaks_matched'],
            'total_peaks_ideal': metrics['total_peaks_ideal'],
            'total_patterns': metrics['total_patterns']
        }
    
    return final_grouped_stats

def print_cumulative_stats(stats: Dict[str, Dict[str, float]], sort_by: str = 'avg_ssim', group_by_model: bool = True):
    """
    Print cumulative statistics in a formatted table, sorted by a specified metric.
    Can group statistics by model type.
    
    Args:
        stats (Dict[str, Dict[str, float]]): Dictionary of statistics per model
        sort_by (str): Metric to sort by
        group_by_model (bool): Whether to group statistics by model type
    """
    if group_by_model:
        stats = group_stats_by_model_type(stats)
    
    # Convert to DataFrame for easier formatting
    rows = []
    for model, metrics in stats.items():
        row = {'Model': model}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values(by=sort_by, ascending=False)
    
    # Format the metrics for better readability
    formatted_df = df.copy()
    for col in df.columns:
        if col == 'Model':
            continue
        if col in ['total_peaks_matched', 'total_peaks_ideal', 'total_patterns']:
            formatted_df[col] = df[col].map(lambda x: f"{int(x):,}")
        else:
            formatted_df[col] = df[col].map(lambda x: f"{x:.4f}")
    
    # Print formatted table with a title indicating grouping
    print("\nCumulative Statistics {} (sorted by {}):\n".format(
        "Grouped by Model Type" if group_by_model else "Per Model and Iteration",
        sort_by
    ))
    print(formatted_df.to_string())
    
    # Print summary statistics
    if group_by_model:
        print("\nSummary:")
        print(f"Total number of model types: {len(df)}")
        print(f"Total patterns processed: {df['total_patterns'].astype(int).sum():,}")
        print(f"Total peaks detected: {df['total_peaks_matched'].astype(int).sum():,} / {df['total_peaks_ideal'].astype(int).sum():,}")
        print(f"Overall peak detection rate: {df['total_peaks_matched'].astype(int).sum() / df['total_peaks_ideal'].astype(int).sum():.4f}")

def create_stats_table_figure(stats: Dict[str, Dict[str, float]], 
                           sort_by: str = 'avg_ssim',
                           group_by_model: bool = True,
                           figsize: Tuple[float, float] = (15, 5)) -> plt.Figure:
    """
    Create a formatted table figure from the statistics.
    
    Args:
        stats (Dict[str, Dict[str, float]]): Dictionary of statistics per model
        sort_by (str): Metric to sort by
        group_by_model (bool): Whether to group statistics by model type
        figsize (Tuple[float, float]): Figure size in inches
        
    Returns:
        plt.Figure: The created figure
    """
    if group_by_model:
        stats = group_stats_by_model_type(stats)
    
    # Convert to DataFrame and sort
    rows = []
    for model, metrics in stats.items():
        row = {'Model': model}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values(by=sort_by, ascending=False)
    
    # Select and rename columns for display
    columns_to_show = [
        'Model',
        'avg_psnr',
        'avg_ssim',
        'peak_detection_rate',
        'avg_peak_dist',
        'avg_fwhm_diff',
        'total_peaks_matched',
        'total_peaks_ideal',
        'pattern_count' if 'pattern_count' in df.columns else 'total_patterns'
    ]
    
    column_labels = {
        'Model': 'Model Type',
        'avg_psnr': 'PSNR (dB)',
        'avg_ssim': 'SSIM',
        'peak_detection_rate': 'Peak Detection Rate',
        'avg_peak_dist': 'Avg Peak Distance',
        'avg_fwhm_diff': 'Avg FWHM Diff',
        'total_peaks_matched': 'Peaks Matched',
        'total_peaks_ideal': 'Total Peaks',
        'pattern_count': 'Patterns',
        'total_patterns': 'Patterns'
    }
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Format the data
    display_df = df[columns_to_show].copy()
    
    # Format numeric columns
    for col in display_df.columns:
        if col == 'Model':
            continue
        if col in ['total_peaks_matched', 'total_peaks_ideal', 'pattern_count', 'total_patterns']:
            display_df[col] = display_df[col].map(lambda x: f"{int(x):,}")
        elif col in ['avg_psnr', 'avg_peak_dist', 'avg_fwhm_diff']:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")
        else:
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    
    # Create table
    table = ax.table(
        cellText=display_df.values,
        colLabels=[column_labels[col] for col in columns_to_show],
        cellLoc='center',
        loc='center',
        colColours=['#E6E6E6'] * len(columns_to_show)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Adjust cell widths based on content
    for (row, col), cell in table.get_celld().items():
        if col == 0:  # Model column
            cell.set_width(0.2)
        else:
            cell.set_width(0.1)
    
    # Add title
    title = f"Model Performance Statistics (Sorted by {column_labels[sort_by]})"
    if group_by_model:
        title += " - Grouped by Model Type"
    #plt.title(title, pad=20, size=12, weight='bold')
    
    # Add summary statistics as text below the table
    if group_by_model:
        patterns_col = 'pattern_count' if 'pattern_count' in df.columns else 'total_patterns'
        summary_text = [
            f"Total number of model types: {len(df)}",
            f"Total patterns processed: {df[patterns_col].astype(int).sum():,}",
            f"Total peaks detected: {df['total_peaks_matched'].astype(int).sum():,} / {df['total_peaks_ideal'].astype(int).sum():,}",
            f"Overall peak detection rate: {df['total_peaks_matched'].astype(int).sum() / df['total_peaks_ideal'].astype(int).sum():.4f}"
        ]
        #plt.figtext(0.1, 0.02, '\n'.join(summary_text), fontsize=9, va='bottom')
    
    plt.tight_layout()
    return fig

def save_stats_table(stats: Dict[str, Dict[str, float]], 
                    save_path: str,
                    sort_by: str = 'avg_ssim',
                    group_by_model: bool = True,
                    figsize: Tuple[float, float] = (15, 5)):
    """
    Create and save a formatted table of statistics as a figure.
    
    Args:
        stats (Dict[str, Dict[str, float]]): Dictionary of statistics per model
        save_path (str): Path where to save the figure
        sort_by (str): Metric to sort by
        group_by_model (bool): Whether to group statistics by model type
        figsize (Tuple[float, float]): Figure size in inches
    """
    fig = create_stats_table_figure(stats, sort_by, group_by_model, figsize)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

# Example usage:
# save_stats_table(stats, 'model_stats.pdf', sort_by='peak_detection_rate')
# # Or to try different sortings:
# for metric in ['avg_ssim', 'avg_psnr', 'peak_detection_rate']:
#     save_stats_table(stats, f'model_stats_{metric}.pdf', sort_by=metric)

#%%
# Example usage:
model_configs = {
    'iterations': [500],#[2, 10, 25, 50, 100, 500],
    'models': {
        'L1_no_Unet': 'best_model_ZCB_9_no_Unet_epoch_{}.pth',
        'L1_Unet': 'best_model_ZCB_9_Unet_epoch_{}.pth',
        'L2_no_Unet': 'best_model_ZCB_9_32_no_Unet_epoch_{}_L2.pth',
        'L2_Unet': 'best_model_ZCB_9_32_Unet_epoch_{}_L2.pth',
        'pearson_no_Unet': 'best_model_ZCB_9_no_Unet_epoch_{}_pearson_loss.pth',
        'pearson_Unet': 'best_model_ZCB_9_Unet_epoch_{}_pearson_loss.pth',
        #'pearson_no_Unet': 'best_model_ZCB_9_31_no_Unet_epoch_{}_pearson_loss.pth',
        #'pearson_Unet': 'best_model_ZCB_9_31_Unet_epoch_{}_pearson_loss.pth'
    }
}
#%%
# Load the input data
mask = np.load('/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy')

#%%
ind=random.randint(0,10800)
ind=4111#8370#2362#8370#4111#9375#338#5840
print(f'Using index {ind}')
# preprocess diffraction pattern
#dp_pp,_,_ = ptNN_U.preprocess_ZCB_9(np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/32/output_hanning_conv_{ind:05d}.npz')['convDP'],mask)
#dp_pp_IDEAL,_,_ = ptNN_U.preprocess_ZCB_9(np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/32/output_hanning_conv_{ind:05d}.npz')['pinholeDP_extra_conv'],mask=np.ones(dp_pp[0][0].shape))
hr,kr,lr=3,1,0
dp_pp,_,_ = ptNN_U.preprocess_ZCB_9(np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls400_gs1024_lsp6_r3.0_typeSC/output_hanning_conv_{hr}_{kr}_{lr}_00006.npz')['convDP'],mask)
dp_pp_IDEAL,_,_ = ptNN_U.preprocess_ZCB_9(np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/lattice_ls400_gs1024_lsp6_r3.0_typeSC/output_hanning_conv_{hr}_{kr}_{lr}_00006.npz')['pinholeDP_extra_conv'],mask=np.ones(dp_pp[0][0].shape))

#%%

# #Experimental data
# scan_name = 'ZCB_9_3D'
# scan_id = '5065'
# base_path = f'/net/micdata/data2/12IDC/2025_Feb/ptycho/{scan_id}'
# #scan informtion
# ncols=36
# nrows=29
# center=(517,575)
# # load ZCB_9_3D diffraction data
# base_path="/net/micdata/data2/12IDC/2025_Feb/ptycho/"
# scan_numbers=[int(scan_id)]#[5101]#[5065]#get_sample_scans(scan_info, ' ZCB_9_3D')['scan_number'].values#[5045,5065,5102,5150]
# all_dps=[]
# for scan_number in scan_numbers:
#     print(f"Loading scan {scan_number}")
#     dps=ptNN_U.load_h5_scan_to_npy(base_path, scan_number, plot=False,point_data=True)
#     all_dps.append(dps)
# all_dps=np.asarray(all_dps)

# #%%
# #crop diffraction patterns
# dpsize=256
# ri=660
# dp=dps[ri][center[0]-dpsize//2:center[0]+dpsize//2,
#     center[1]-dpsize//2:center[1]+dpsize//2]
# dp_pp,_,_ = ptNN_U.preprocess_ZCB_9(dp,mask)
# dp_pp_IDEAL,_,_ = ptNN_U.preprocess_ZCB_9(np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/32/output_hanning_conv_{ind:05d}.npz')['pinholeDP_extra_conv'],mask=np.ones(dp_pp[0][0].shape))


fig,ax = plt.subplots(1,2)
im1=ax[0].imshow(dp_pp.squeeze().cpu().numpy())
im2=ax[1].imshow(dp_pp_IDEAL.squeeze().cpu().numpy())
plt.colorbar(im1,ax=ax[0])
plt.colorbar(im2,ax=ax[1])
ax[0].set_title('Convolution')
ax[1].set_title('Ideal')
plt.show()


# Create the comparison grid
fig = create_comparison_grid_from_config(
    model_configs=model_configs,
    input_data=dp_pp.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float),
    ideal_data=dp_pp_IDEAL.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float),
    base_path="/net/micdata/data2/12IDC/ptychosaxs/models/ZCB_9_3D/",
    figsize=(15, 15),
    calculate_psnr=True,
    calculate_ssim=True,
    calculate_xcorr=False,
    calculate_peaks=True
)

#%%
save_comparison_grid(fig, 'comparison_grid.pdf')

# %%
# Define your list of indices
indices_list = [
    (1,0,0),
    (1,1,1),
    (2,1,1),
    (3,1,0),
    (3,2,1),
    (2,0,0),
    (2,2,0)
    # ... add more combinations as needed
]

# Calculate cumulative stats
stats = calculate_cumulative_stats(
    model_configs=model_configs,
    indices_list=indices_list,
    base_path="/net/micdata/data2/12IDC/ptychosaxs/models/ZCB_9_3D/",
    calculate_psnr=True,
    calculate_ssim=True,
    calculate_xcorr=False,
    calculate_peaks=True,
    calculate_peak_sensitivity=False,
    peak_sigma=0.714,
    central_only=False
)

# Print stats sorted by different metrics
print_cumulative_stats(stats, sort_by='avg_ssim')  # Sort by SSIM
print_cumulative_stats(stats, sort_by='peak_detection_rate')  # Sort by peak detection rate
print_cumulative_stats(stats, sort_by='avg_psnr')  # Sort by PSNR




# %%
create_stats_table_figure(stats, sort_by='peak_detection_rate', group_by_model=True)
plt.show()
# %%
save_stats_table(stats, 'model_stats.pdf', sort_by='peak_detection_rate', group_by_model=True)