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

def create_row_col_mask(image_shape, rows_to_mask=None, cols_to_mask=None):
    """
    Creates a mask with specified rows and columns set to zero.

    Parameters:
    - image_shape (tuple): Shape of the 2D array for which the mask is created.
    - rows_to_mask (list of int, optional): Indices of rows to set to zero.
    - cols_to_mask (list of int, optional): Indices of columns to set to zero.

    Returns:
    - mask (2D numpy array): The mask with `1` in valid regions and `0` in the masked rows/columns.
    """
    # Initialize the mask with ones
    mask = np.ones(image_shape, dtype=np.float32)

    # Mask rows
    if rows_to_mask:
        mask[rows_to_mask, :] = 0

    # Mask columns
    if cols_to_mask:
        mask[:, cols_to_mask] = 0

    return mask

    # Example usage
    #
    # X:\deconvolutionNN\data\mask\mask_CindyPtychoshelvesHDF5preprocessed.npy
    # temp=np.random.rand(conv_DPs[0].shape[0],conv_DPs[0].shape[1])
    # fig,ax=plt.subplots(1,3,figsize=(15,5))
    # ax[0].imshow(temp)
    # mask=create_row_col_mask(temp.shape, rows_to_mask=[range(0,8),range(247,255)], cols_to_mask=[range(0,8),range(247,255)])
    # ax[1].imshow(mask)
    # result=apply_mask(temp,mask)
    # ax[2].imshow(result)
    # plt.show()
    # np.save(os.path.abspath(os.path.join(os.getcwd(), '../../data/mask/mask_CindyPtychoshelvesHDF5preprocessed.npy')),mask)
    # 
    # X:\deconvolutionNN\data\mask\mask_Chansong256x256Cropped.npy
    # temp=np.random.rand(conv_DPs[0].shape[0],conv_DPs[0].shape[1])
    # lbound,ubound=(23,38),(235,250)
    # fig,ax=plt.subplots(1,3,figsize=(15,5))
    # ax[0].imshow(temp)
    # mask=create_row_col_mask(temp.shape, rows_to_mask=[range(23,38),range(235,250)], cols_to_mask=None)
    # ax[1].imshow(mask)
    # result=apply_mask(temp,mask)
    # ax[2].imshow(result)
    # plt.show()
    # np.save(os.path.abspath(os.path.join(os.getcwd(), '../../data/mask/mask_Chansong256x256Cropped.npy')),mask)

def apply_mask(image, mask):
    """
    Applies the given mask to an image, setting masked regions to the minimum value of the image.

    Parameters:
    - image (2D numpy array): The input image to which the mask is applied.
    - mask (2D numpy array): The mask to apply (same shape as the image).

    Returns:
    - masked_image (2D numpy array): The masked image.
    """
    if image.shape != mask.shape:
        raise ValueError("Image and mask shapes do not match.")
    
    # Create a copy of the image
    masked_image = image.copy()
    
    # Set masked areas (where mask == 0) to the minimum value of the image
    min_value = image.min()
    masked_image[mask == 0] = min_value

    return masked_image

def log10_custom(arr):
    # Create a mask for positive values
    positive_mask = arr > 0
    
    # Initialize result array
    result = np.zeros_like(arr, dtype=float)
    
    # Calculate log10 only on positive values
    log10_positive = np.log10(arr[positive_mask])
    
    # Find the minimum log10 value from the positive entries
    min_log10_value = log10_positive.min() if log10_positive.size > 0 else 0
    
    # Set positive entries to their log10 values
    result[positive_mask] = log10_positive
    
    # Set non-positive entries to the minimum log10 value
    result[~positive_mask] = min_log10_value
    
    return result

    '''
    Explanation
    We calculate the log10 values only on positive entries.
    min_log10_value holds the minimum log10 value among positive entries.
    We assign min_log10_value to any non-positive entries in the array, so negative or zero values get this minimum value.
    This approach ensures the result is smoothly scaled with log10 values, using the minimum log10 value for non-positive entries. Let me know if you'd like to test it!
    '''
    
def scale_back(dp,scale_factor,constant):
    '''
    Scale back normalized intensity patterns used in the neural network training (scale_factor and constant plus a inverse log10 operation)
    '''
    temp=dp*scale_factor + constant
    temp=10**(temp)
    return temp