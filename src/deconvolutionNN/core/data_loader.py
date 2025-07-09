"""Data loading and preprocessing utilities for deconvolution."""

from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from skimage.transform import resize
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def load_convoluted_and_ideal_patterns(
    h5_file_path: Union[str, Path], max_dps: int = 10800
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load convoluted and ideal diffraction patterns from HDF5 file.

    This function loads three types of diffraction patterns:
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
    print("Loading data...")

    with h5py.File(h5_file_path, "r") as h5f:
        # Get the keys (dataset names)
        dataset_keys = list(h5f.keys())
        num_datasets = len(dataset_keys) // 3  # Assuming convDP and pinholeDP pairs
        print(f"{num_datasets} diffraction patterns available")

        # Initialize empty lists for the data
        conv_DPs = []
        pinhole_DPs = []

        # Number of diffraction patterns to actually load
        numDPs = min(max_dps, num_datasets)

        # Use tqdm for progress tracking
        for i in tqdm(range(num_datasets)[:numDPs], desc="Loading HDF5 datasets"):
            conv_DPs.append(h5f[f"convDP_{i}"][:])  # Load convDP dataset
            pinhole_DPs.append(h5f[f"pinholeDP_{i}"][:])  # Load pinholeDP dataset

    # Convert to numpy arrays
    conv_DPs = np.asarray(conv_DPs)
    ideal_DPs = np.asarray(pinhole_DPs)
    probe_DPs = np.ones(conv_DPs.shape)  # Dummy array for testing network with a probe

    print(f"Loaded {len(conv_DPs)} diffraction patterns")
    print(f"Convoluted patterns shape: {conv_DPs.shape}")
    print(f"Ideal patterns shape: {ideal_DPs.shape}")
    print(f"Probe patterns shape: {probe_DPs.shape}")

    return conv_DPs, ideal_DPs, probe_DPs


def load_probe_kernel(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load probe kernel from HDF5 file.

    Args:
        file_path: Path to the HDF5 file containing the probe

    Returns:
        Complex probe kernel array
    """
    with h5py.File(file_path, "r") as f:
        probe = f["probe"][0][0]
    return probe


def resize_probe(probe: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize probe kernel while preserving complex structure.

    Args:
        probe: Complex probe kernel array
        target_size: Target size for both dimensions

    Returns:
        Resized complex probe kernel
    """
    # Separate resize for real and imaginary components
    probe_real = resize(
        np.real(probe),
        (target_size, target_size),
        preserve_range=True,
        anti_aliasing=True,
    )
    probe_imag = resize(
        np.imag(probe),
        (target_size, target_size),
        preserve_range=True,
        anti_aliasing=True,
    )

    # Recombine into complex array
    return probe_real + 1j * probe_imag


def load_diffraction_patterns(
    data_dir: Union[str, Path], scan_numbers: list[int]
) -> np.ndarray:
    """
    Load diffraction patterns from multiple scans.

    Args:
        data_dir: Directory containing scan data
        scan_numbers: List of scan numbers to load

    Returns:
        Array of diffraction patterns
    """
    all_dps = []
    for scan in scan_numbers:
        # Note: This assumes ptNN_U.load_h5_scan_to_npy exists
        # You may need to implement this or use a different loading method
        try:
            dps = load_h5_scan_to_npy(Path(data_dir), scan, plot=False)
            all_dps.append(dps)
        except ImportError:
            # Fallback: implement basic loading here
            dps = load_scan_data_fallback(Path(data_dir), scan)
            all_dps.append(dps)

    temp_dps = np.asarray(all_dps)
    return temp_dps.reshape(-1, temp_dps.shape[2], temp_dps.shape[3])


def load_scan_data_fallback(data_dir: Path, scan_number: int) -> np.ndarray:
    """
    Fallback method to load scan data when ptNN_U is not available.

    Args:
        data_dir: Directory containing scan data
        scan_number: Scan number to load

    Returns:
        Array of diffraction patterns
    """
    # This is a placeholder implementation
    # You should replace this with your actual data loading logic
    scan_file = data_dir / f"scan_{scan_number:04d}.h5"

    if not scan_file.exists():
        raise FileNotFoundError(f"Scan file not found: {scan_file}")

    with h5py.File(scan_file, "r") as f:
        # Adjust this based on your actual data structure
        if "data" in f:
            return f["data"][:]
        elif "diffraction_patterns" in f:
            return f["diffraction_patterns"][:]
        else:
            # Try to find any dataset
            key = list(f.keys())[0]
            return f[key][:]


def create_center_mask(shape: tuple[int, int], center_radius: int = 40) -> np.ndarray:
    """
    Create a mask that excludes the central beam region.

    Args:
        shape: Shape of the mask (height, width)
        center_radius: Radius of the central beam to mask

    Returns:
        Boolean mask (True for pixels to keep)
    """
    y, x = np.ogrid[: shape[0], : shape[1]]
    center_y, center_x = shape[0] // 2, shape[1] // 2

    # Distance from center for each pixel
    dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Create mask (True for pixels we want to keep)
    return dist_from_center > center_radius


def preprocess_diffraction_patterns(
    dps: np.ndarray,
    center: np.ndarray,
    target_size: int,
    center_radius: int = 40,
    min_intensity_threshold: float = 10000.0,
) -> np.ndarray:
    """
    Preprocess diffraction patterns for training.

    Args:
        dps: Raw diffraction patterns
        center: Center coordinates for cropping
        target_size: Target size for resizing
        center_radius: Radius for central beam masking
        min_intensity_threshold: Minimum intensity threshold for filtering

    Returns:
        Preprocessed diffraction patterns
    """
    print("Shuffling indices")
    indices = np.arange(dps.shape[0])
    np.random.shuffle(indices)

    print("Shuffling diffraction patterns")
    dps_shuff = dps[indices]

    print("Applying log10 transformation")
    # Apply log10 transformation (you may need to implement log10_custom)
    try:
        amp_dps = log10_custom(dps_shuff)
    except NameError:
        # Fallback to numpy log10
        amp_dps = np.log10(dps_shuff + 1e-10)

    print("Resizing")
    # Crop and resize
    crop_size = target_size
    amp_dps_red = np.asarray(
        [
            resize(
                d[
                    center[0] - crop_size // 2 : center[0] + crop_size // 2,
                    center[1] - crop_size // 2 : center[1] + crop_size // 2,
                ],
                (target_size, target_size),
                preserve_range=True,
                anti_aliasing=True,
            )
            for d in tqdm(amp_dps)
        ]
    )

    print("Normalizing")
    # Normalize each pattern
    amp_dps_red = np.asarray(
        [(a - np.min(a)) / (np.max(a) - np.min(a)) for a in tqdm(amp_dps_red)]
    )

    # Filter patterns based on intensity
    mask = create_center_mask((target_size, target_size), center_radius)
    filtered_dps = []

    for dp in amp_dps_red:
        total_intensity = np.sum(dp * mask)
        if total_intensity > min_intensity_threshold:
            filtered_dps.append(dp)

    print(f"{len(filtered_dps)} total diffraction patterns after filtering")
    return np.asarray(filtered_dps)


def create_data_loaders(
    data: np.ndarray,
    batch_size: int,
    train_split: float = 0.75,
    val_split: float = 0.125,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        data: Preprocessed diffraction patterns
        batch_size: Batch size for training
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    n_total = data.shape[0]
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    # Reshape data for PyTorch: (N, 1, H, W)
    data_tensor = torch.Tensor(data.reshape(-1, 1, data.shape[1], data.shape[2]))

    # Shuffle data
    data_tensor = torch.Tensor(shuffle(data_tensor.numpy(), random_state=0))

    # Create datasets
    train_data = TensorDataset(data_tensor[:n_train])
    val_data = TensorDataset(data_tensor[n_train : n_train + n_val])
    test_data = TensorDataset(data_tensor[n_train + n_val :])

    # Create data loaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# Placeholder functions for compatibility
def log10_custom(data: np.ndarray) -> np.ndarray:
    """
    Custom log10 transformation for diffraction patterns.

    Args:
        data: Input data

    Returns:
        Log10 transformed data
    """
    # This should be implemented based on your specific needs
    return np.log10(data + 1e-10)


def load_h5_scan_to_npy(
    data_dir: Path, scan_number: int, plot: bool = False
) -> np.ndarray:
    """
    Load H5 scan data to numpy array.

    Args:
        data_dir: Directory containing scan data
        scan_number: Scan number to load
        plot: Whether to plot the data

    Returns:
        Array of diffraction patterns
    """
    # This is a placeholder - implement based on your actual data structure
    scan_file = data_dir / f"scan_{scan_number:04d}.h5"

    if not scan_file.exists():
        raise FileNotFoundError(f"Scan file not found: {scan_file}")

    with h5py.File(scan_file, "r") as f:
        # Adjust this based on your actual data structure
        if "data" in f:
            return f["data"][:]
        elif "diffraction_patterns" in f:
            return f["diffraction_patterns"][:]
        else:
            # Try to find any dataset
            key = list(f.keys())[0]
            return f[key][:]
