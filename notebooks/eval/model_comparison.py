#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from typing import List, Dict, Tuple
import pandas as pd
import sys
import os
from matplotlib import colors
from skimage.metrics import structural_similarity as ssim
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

def calculate_metrics(img1: torch.Tensor, img2: torch.Tensor) -> Tuple[float, float]:
    """
    Calculate PSNR and SSIM between two images.
    Images are normalized before metric calculation.
    
    Args:
        img1 (torch.Tensor): First image
        img2 (torch.Tensor): Second image
        
    Returns:
        Tuple[float, float]: PSNR and SSIM values
    """
    # Normalize both images
    img1_norm = normalize_image(img1)
    img2_norm = normalize_image(img2)
    
    # Convert to numpy arrays
    img1_np = img1_norm.squeeze().cpu().numpy()
    img2_np = img2_norm.squeeze().cpu().numpy()
    
    # Calculate MSE and PSNR
    mse = np.mean((img1_np - img2_np) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 1.0  # Now using normalized range [0, 1]
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    # Calculate SSIM
    ssim_value = ssim(img1_np, img2_np, data_range=1.0)
    
    return psnr, ssim_value

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
                                     figsize: Tuple[int, int] = (20, 15)) -> plt.Figure:
    """
    Create a grid plot comparing outputs from different models using model_configs.
    Shows input and ideal images above the model comparison grid.
    Includes PSNR and SSIM values for each model output compared to the ideal image.
    
    Args:
        model_configs (Dict): Dictionary containing model configurations
        input_data (torch.Tensor): Input data to run through the models
        ideal_data (torch.Tensor): Ideal/ground truth data for comparison
        base_path (str): Base path where models are stored
        figsize (Tuple[int, int]): Figure size
        
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
    
    # Create figure with extra row for input and ideal images
    fig = plt.figure(figsize=figsize)
    
    # Create a grid for the entire figure
    gs = fig.add_gridspec(nrows + 1, ncols + 2, height_ratios=[1] + [1]*nrows)
    
    # Plot input and ideal images in the top row
    ax_input = fig.add_subplot(gs[0, 0])
    ax_ideal = fig.add_subplot(gs[0, 1])
    
    # Plot input image
    im_input = ax_input.imshow(input_data.squeeze().cpu().numpy(), cmap='jet')
    ax_input.set_title('Input Image')
    plt.colorbar(im_input, ax=ax_input)
    
    # Plot ideal image
    im_ideal = ax_ideal.imshow(ideal_data.squeeze().cpu().numpy(), cmap='jet')
    ax_ideal.set_title('Ideal Image')
    plt.colorbar(im_ideal, ax=ax_ideal)
    
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
        
        # Calculate metrics
        psnr, ssim_value = calculate_metrics(output, ideal_data)
        
        # Plot
        ax = axes[row_idx, col_idx]
        im = ax.imshow(output.squeeze().cpu().numpy(), cmap='jet')
        plt.colorbar(im, ax=ax)
        
        # Add metrics text in the corner
        metrics_text = f'PSNR: {psnr:.2f} dB\nSSIM: {ssim_value:.4f}'
        ax.text(0.02, 0.98, metrics_text,
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
# 'L2_no_Unet': 'best_model_ZCB_9_no_Unet_epoch_{}_L2.pth',
# 'L2_Unet': 'best_model_ZCB_9_32_Unet_epoch_{}_L2.pth',
# Example usage:
model_configs = {
    'iterations': [2, 10, 25, 50, 100, 500],
    'models': {
        'L1_no_Unet': 'best_model_ZCB_9_no_Unet_epoch_{}.pth',
        'L1_Unet': 'best_model_ZCB_9_Unet_epoch_{}.pth',
        'pearson_no_Unet': 'best_model_ZCB_9_no_Unet_epoch_{}_pearson_loss.pth',
        'pearson_Unet': 'best_model_ZCB_9_Unet_epoch_{}_pearson_loss.pth'
    }
}

# Load the input data
mask = np.load('/home/beams/PTYCHOSAXS/deconvolutionNN/data/mask/mask_ZCB_9_3D.npy')
ind=random.randint(0,10800)
ind=338
print(f'Using index {ind}')
dp_pp,_,_ = ptNN_U.preprocess_ZCB_9(np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/32/output_hanning_conv_{ind:05d}.npz')['convDP'],mask)
dp_pp_IDEAL,_,_ = ptNN_U.preprocess_ZCB_9(np.load(f'/net/micdata/data2/12IDC/ptychosaxs/data/diff_sim/32/output_hanning_conv_{ind:05d}.npz')['pinholeDP_extra_conv'],mask=np.ones(dp_pp[0][0].shape))
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
    figsize=(20, 15)
)
# %%
