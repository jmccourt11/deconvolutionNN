import os
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

def read_hdf5_file(file_path):    
    """
    Reads an HDF5 file and returns its contents.

    Parameters:
    file_path (str): The path to the HDF5 file.

    Returns:
    dict: A dictionary with dataset names as keys and their data as values.
    """
    data_dict = {}

    try:
        with h5py.File(file_path, 'r') as hdf_file:
            def extract_data(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data_dict[name] = obj[()]

            hdf_file.visititems(extract_data)
    except Exception as e:
        print(f"An error occurred: {e}")

    return data_dict
def find_directories_with_number(base_path, number):
    """
    Finds immediate subdirectories containing a specific number in their name,
    allowing for flexible number formatting.

    Args:
    - base_path (str): The path to the directory to search.
    - number (int): The number to search for in subdirectory names.

    Returns:
    - list: A list of matching directory paths.
    """
    matching_dirs = []
    # Create a regex pattern to match the number with optional leading zeros anywhere in the name
    #number_pattern = rf"0*{number}\b"
    number_pattern = rf"(^|[^0-9])0*{number}([^0-9]|$)"

    try:
        # List only directories in the base path
        for entry in os.listdir(base_path):
            full_path = os.path.join(base_path, entry)
            # Check if the entry is a directory and matches the pattern
            if os.path.isdir(full_path) and re.search(number_pattern, entry):
                matching_dirs.append(full_path)
    except FileNotFoundError:
        print(f"The path '{base_path}' does not exist.")
    except PermissionError:
        print(f"Permission denied to access '{base_path}'.")

    return [Path(m) for m in matching_dirs]
def load_h5_scan_to_npy(file_path,scan,plot=True):
    # For loading cindy ptycho scan data
    # file_path = '/net/micdata/data2/12IDC/2021_Nov/ptycho/'
    # scan = 1125 (e.g.)
    dps=[]
    file_path_new=find_directories_with_number(file_path,scan)[0]
    for filename in os.listdir(file_path_new)[:-1]:
        filename = file_path_new / filename
        data = read_hdf5_file(filename)['entry/data/data']
        print(filename)
        for j in range(0,len(data)):
            dps.append(data[j])
            if plot:
                plt.figure()
                plt.imshow(data[j],norm=colors.LogNorm())
                plt.show()
    dps=np.asarray(dps)
    return dps