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
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../../NN/ptychosaxsNN/')))
import utils.ptychosaxsNN_utils as ptNN_U
import importlib
importlib.reload(ptNN_U)
import random
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

def find_peaks_and_fwhm(image: np.ndarray, threshold: float = 0.25, sigma: float = 1.0) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Find peaks and their FWHM in a 2D image.
    
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
    
    # Find peaks above threshold
    for i in range(1, smoothed.shape[0]-1):
        for j in range(1, smoothed.shape[1]-1):
            if smoothed[i,j] > threshold_val:
                if (smoothed[i,j] > smoothed[i-1,j] and 
                    smoothed[i,j] > smoothed[i+1,j] and
                    smoothed[i,j] > smoothed[i,j-1] and
                    smoothed[i,j] > smoothed[i,j+1]):
                    peaks.append((i, j))
                    
                    # Calculate FWHM in x and y directions
                    x_profile = smoothed[i,:]
                    y_profile = smoothed[:,j]
                    
                    # Find half-max points
                    half_max = smoothed[i,j] / 2
                    x_fwhm = np.sum(x_profile > half_max)
                    y_fwhm = np.sum(y_profile > half_max)
                    
                    # Use average of x and y FWHM
                    fwhm_values.append((x_fwhm + y_fwhm) / 2)
    
    return peaks, fwhm_values

def calculate_metrics(img1: torch.Tensor, 
                     img2: torch.Tensor,
                     calculate_psnr: bool = True,
                     calculate_ssim: bool = True,
                     calculate_xcorr: bool = False,
                     calculate_peaks: bool = True) -> Dict[str, float]:
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
                                     peak_sigma: float = 2.0) -> plt.Figure:
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
                metrics_text.append(f'Sigma: {peak_sigma:.1f}')
        
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

#%%
# Example usage:
model_configs = {
    'iterations': [2, 10, 25, 50, 100, 500],
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

# Load the input data
mask = np.load('/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy')
ind=random.randint(0,10800)
ind=4111#9375#338#5840
print(f'Using index {ind}')
# preprocess diffraction pattern
dp_pp,_,_ = ptNN_U.preprocess_ZCB_9(np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/32/output_hanning_conv_{ind:05d}.npz')['convDP'],mask)
dp_pp_IDEAL,_,_ = ptNN_U.preprocess_ZCB_9(np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/32/output_hanning_conv_{ind:05d}.npz')['pinholeDP_extra_conv'],mask=np.ones(dp_pp[0][0].shape))


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
# %%
