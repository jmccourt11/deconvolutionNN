import numpy as np


def replace_2d_array_values_by_row_indices(array, start, end):
    """
    Replaces values in a 2D numpy array with 0 if their row indices fall within the specified range.

    Parameters:
    array (np.ndarray): Input 2D numpy array.
    start (int): Start index of the range (inclusive).
    end (int): End index of the range (inclusive).

    Returns:
    np.ndarray: 2D numpy array with specified rows replaced with 0.

    Example usage:
    array = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12],
                      [13, 14, 15]])
    
    modified_array = replace_2d_array_values_by_row_indices(array, 1, 3)
    print(modified_array)

    """
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("The input must be a 2D numpy array")
    
    if not (0 <= start <= end < array.shape[0]):
        raise ValueError("Invalid range specified")

    array[start:end+1, :] = np.min(array)
    return array

def replace_2d_array_values_by_column_indices(array, start, end):
    """
    Replaces values in a 2D numpy array with 0 if their row indices fall within the specified range.

    Parameters:
    array (np.ndarray): Input 2D numpy array.
    start (int): Start index of the range (inclusive).
    end (int): End index of the range (inclusive).

    Returns:
    np.ndarray: 2D numpy array with specified rows replaced with 0.

    Example usage:
    array = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [10, 11, 12],
                      [13, 14, 15]])
    
    modified_array = replace_2d_array_values_by_row_indices(array, 1, 3)
    print(modified_array)

    """
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("The input must be a 2D numpy array")
    
    if not (0 <= start <= end < array.shape[0]):
        raise ValueError("Invalid range specified")

    array[:,start:end+1] = np.min(array)
    return array

def create_circular_mask(image,radius=48):
    """
    Creates a circular mask at the center of the image.
    """
    # Get the dimensions of the image
    h, w = image.shape[:2]

    # Center
    center_x, center_y = w // 2, h // 2
    
    # Create a grid of x and y coordinates
    y, x = np.ogrid[:h, :w]
    
    # Calculate the distance from each pixel to the center
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    # Create the circular mask
    mask = distance_from_center >= radius
    
    return mask.astype(np.uint8)  # Return as uint8 (0s and 1s)