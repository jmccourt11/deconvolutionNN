#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors
import sys
import os
import importlib
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../NN/ptychosaxsNN/'))) 
import utils.ptychosaxsNN_utils as ptNN_U
import ptychosaxsNN.ptychosaxsNN as ptNN
importlib.reload(ptNN_U)
importlib.reload(ptNN)



# Function to perform circular average
def circular_average(image, center, radius, num_points=360):
    # Create coordinate grids
    y, x = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    
    # Calculate distance from center for each pixel
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Create mask for pixels at specified radius (with small tolerance)
    tolerance = 0.5
    mask = (r >= radius - tolerance) & (r <= radius + tolerance)
    
    # Get values of pixels on the circle
    values = image[mask]
    
    # Calculate angles for pixels on the circle
    y_masked = y[mask]
    x_masked = x[mask]
    angles = np.arctan2(y_masked - center[0], x_masked - center[1])
    
    return np.mean(values)

def azimuthal_average(image, center, max_radius=None, num_bins=100):
    """
    Calculate 1D azimuthal average of a 2D image.
    
    Parameters:
    -----------
    image : 2D numpy array
        Input image/diffraction pattern
    center : tuple
        (y, x) coordinates of the center point
    max_radius : int, optional
        Maximum radius to consider. If None, uses the maximum possible radius
    num_bins : int, optional
        Number of radial bins to use
        
    Returns:
    --------
    radii : 1D numpy array
        Radial distances for each bin
    intensity : 1D numpy array
        Average intensity at each radius
    """
    # Create coordinate grids
    y, x = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
    
    # Calculate distance from center for each pixel
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Set maximum radius if not provided
    if max_radius is None:
        max_radius = np.max(r)
    
    # Create radial bins
    bins = np.linspace(0, max_radius, num_bins + 1)
    radii = (bins[1:] + bins[:-1]) / 2  # Center of each bin
    
    # Calculate average intensity in each bin
    intensity = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if np.any(mask):
            intensity[i] = np.mean(image[mask])
    
    return radii, intensity

def circular_profile(image, center, radius, num_points=360):
    """
    Extract intensity values along a circle and return as a line plot.
    
    Parameters:
    -----------
    image : 2D numpy array
        Input image/diffraction pattern
    center : tuple
        (y, x) coordinates of the center point
    radius : float
        Radius of the circle
    num_points : int, optional
        Number of points to sample along the circle
        
    Returns:
    --------
    angles : 1D numpy array
        Angles in radians
    intensities : 1D numpy array
        Intensity values at each angle
    """
    # Create angles for sampling
    angles = np.linspace(0, 2*np.pi, num_points)
    
    # Calculate points along the circle
    x = center[1] + radius * np.cos(angles)
    y = center[0] + radius * np.sin(angles)
    
    # Convert to integer indices
    x_idx = np.round(x).astype(int)
    y_idx = np.round(y).astype(int)
    
    # Ensure indices are within bounds
    mask = (x_idx >= 0) & (x_idx < image.shape[1]) & (y_idx >= 0) & (y_idx < image.shape[0])
    x_idx = x_idx[mask]
    y_idx = y_idx[mask]
    angles = angles[mask]
    
    # Get intensities
    intensities = image[y_idx, x_idx]
    
    return angles, intensities

#%%

# Create array to store summed diffraction patterns
all_dps = []

# Loop through scans and sum diffraction patterns
for scan in tqdm(np.arange(888,889)):
    try:
        # Load diffraction patterns for this scan
        dps = ptNN_U.load_h5_scan_to_npy(Path(f'/mnt/micdata2/12IDC/2024_Dec/ptycho/'),scan,plot=False)
        # Sum along first axis and append
        all_dps.append(np.sum(dps,axis=0))
    except:
        continue

# Convert to numpy array
summed_dps = np.asarray(all_dps)

#%%
plt.figure(figsize=(10,10))
plt.imshow(summed_dps[0],norm=colors.LogNorm(),cmap='jet')
plt.colorbar()
plt.show()

#%%
center = (734,745)
dp_size=1280
dp=summed_dps[0][center[0]-dp_size//2:center[0]+dp_size//2,center[1]-dp_size//2:center[1]+dp_size//2]

center=dp.shape[0]//2,dp.shape[1]//2

# Create figure with four subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20,20))

# Plot original image
im1 = ax1.imshow(dp, norm=colors.LogNorm(), cmap='jet')
ax1.set_title('Original')
plt.colorbar(im1, ax=ax1)

# Plot image with peak markers and circles
im2 = ax2.imshow(dp, norm=colors.LogNorm(), cmap='jet')
ax2.set_title('With Peak Markers and Radial Circles')
plt.colorbar(im2, ax=ax2)

# Define peak positions and center
peak_positions = [
    (center[0], center[1]),  # Center peak
    (center[0]-128, center[1]),  # Left peak
    (center[0]+128, center[1]),  # Right peak
    (center[0], center[1]-128),  # Bottom peak 
    (center[0], center[1]+128)   # Top peak
]

# Add peak markers
for peak in peak_positions:
    ax2.plot(peak[1], peak[0], 'r+', markersize=15, markeredgewidth=2)



# Calculate and plot azimuthal average
max_radius = 566
radii, intensity = azimuthal_average(dp, center, max_radius=max_radius, num_bins=100)

# Draw multiple circles at different radii
num_circles = 2
circle_radii = np.linspace(456, max_radius, num_circles)
theta = np.linspace(0, 2*np.pi, 100)
for r in circle_radii:
    x = center[1] + r * np.cos(theta)
    y = center[0] + r * np.sin(theta)
    ax2.plot(x, y, 'k--', alpha=0.5, label=f'r={int(r)}')

# Add legend for circles
ax2.legend()

# Plot azimuthal average
ax3.plot(radii, intensity)
ax3.set_title('Azimuthal Average')
ax3.set_xlabel('Radius (pixels)')
ax3.set_ylabel('Average Intensity')
ax3.set_yscale('log')  # Use log scale for intensity

# Add vertical lines at the circle radii
for r in circle_radii:
    ax3.axvline(x=r, color='gray', linestyle='--', alpha=0.5)

# Plot circular profiles for each radius
ax4.set_title('Circular Intensity Profiles')
ax4.set_xlabel('Angle (radians)')
ax4.set_ylabel('Intensity')
ax4.set_yscale('log')

for r in circle_radii:
    angles, intensities = circular_profile(dp, center, r)
    ax4.plot(angles, intensities, label=f'r={int(r)}')

ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()

# %%
