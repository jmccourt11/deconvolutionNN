# 1
"""
Simulates diffraction using form factor (spherical, cubal, icosahedral, octahedral, tetrahedral) and 
structure factor (SC, FCC, BCC) for finite sized nanocrystals
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from scipy.signal import convolve2d as conv2np
from skimage.filters import window


def create_lattice(lattice_type, num_points_x, num_points_y, num_points_z, spacing):
    """
    Create a lattice of points in a 3D space.

    Parameters:
    lattice_type (str): Type of the lattice ('SC', 'BCC', 'FCC').
    num_points_x (int): Number of points along the x-axis.
    num_points_y (int): Number of points along the y-axis.
    num_points_z (int): Number of points along the z-axis.
    spacing (float): Distance between adjacent points.

    Returns:
    np.ndarray: Array of shape (N, 3) containing the lattice points, where N is the total number of points.
    """
    x_coords = np.linspace(0, (num_points_x - 1) * spacing, num_points_x)
    y_coords = np.linspace(0, (num_points_y - 1) * spacing, num_points_y)
    z_coords = np.linspace(0, (num_points_z - 1) * spacing, num_points_z)
    
    xv, yv, zv = np.meshgrid(x_coords, y_coords, z_coords)
    
    if lattice_type == 'SC':
        lattice_points = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
    elif lattice_type == 'BCC':
        lattice_points = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
        xv_bcc = xv + spacing / 2
        yv_bcc = yv + spacing / 2
        zv_bcc = zv + spacing / 2
        bcc_points = np.column_stack([xv_bcc.ravel(), yv_bcc.ravel(), zv_bcc.ravel()])
        lattice_points = np.vstack([lattice_points, bcc_points])
    elif lattice_type == 'FCC':
        lattice_points = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])
        fcc_points = np.vstack([
            np.column_stack([xv.ravel() + spacing / 2, yv.ravel() + spacing / 2, zv.ravel()]),
            np.column_stack([xv.ravel() + spacing / 2, yv.ravel(), zv.ravel() + spacing / 2]),
            np.column_stack([xv.ravel(), yv.ravel() + spacing / 2, zv.ravel() + spacing / 2])
        ])
        lattice_points = np.vstack([lattice_points, fcc_points])
    else:
        raise ValueError("Unsupported lattice type. Supported types are: 'SC', 'BCC', 'FCC'")
    
    return lattice_points
    

def form_factor_sphere(radius, q):
    """
    Form factor for a spherical particle.

    Parameters:
    radius (float): Radius of the sphere.
    q (np.ndarray): The scattering vector.

    Returns:
    float: The form factor value for the given scattering vector.
    """
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return 1.0
    return 3 * (np.sin(q_norm * radius) - q_norm * radius * np.cos(q_norm * radius)) / (q_norm**3 * radius**3)
    
def form_factor_cube(side_length, q):
    """
    Form factor for a cubic particle.

    Parameters:
    side_length (float): Side length of the cube.
    q (np.ndarray): The scattering vector.

    Returns:
    float: The form factor value for the given scattering vector.
    """
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return 1.0
    half_side = side_length / 2
    return np.sinc(q[0] * half_side / np.pi) * np.sinc(q[1] * half_side / np.pi) * np.sinc(q[2] * half_side / np.pi)


def form_factor_icosahedron(edge_length, q):
    """
    Form factor for an icosahedron particle.

    Parameters:
    edge_length (float): Edge length of the icosahedron.
    q (np.ndarray): The scattering vector.

    Returns:
    float: The form factor value for the given scattering vector.
    """
    # Volume of an icosahedron: V = (5/12) * (3 + sqrt(5)) * a^3
    volume = (5/12) * (3 + np.sqrt(5)) * edge_length**3
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return volume
    # Approximate form factor calculation for an icosahedron
    f_q = volume * (np.sin(q_norm * edge_length) / (q_norm * edge_length))
    return f_q
    
def form_factor_octahedron(edge_length, q):
    """
    Form factor for an octahedron particle.

    Parameters:
    edge_length (float): Edge length of the octahedron.
    q (np.ndarray): The scattering vector.

    Returns:
    float: The form factor value for the given scattering vector.
    """
    # Volume of an octahedron: V = (1/3) * sqrt(2) * a^3
    volume = (1/3) * np.sqrt(2) * edge_length**3
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return volume
    # Approximate form factor calculation for an octahedron
    f_q = volume * (np.sin(q_norm * edge_length) / (q_norm * edge_length))
    return f_q
    
def form_factor_tetrahedron(edge_length, q):
    """
    Form factor for a tetrahedron particle.

    Parameters:
    edge_length (float): Edge length of the tetrahedron.
    q (np.ndarray): The scattering vector.

    Returns:
    float: The form factor value for the given scattering vector.
    """
    # Volume of a tetrahedron: V = (1/12) * sqrt(2) * a^3
    volume = (1/12) * np.sqrt(2) * edge_length**3
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return volume
    # Approximate form factor calculation for a tetrahedron
    f_q = volume * (np.sin(q_norm * edge_length) / (q_norm * edge_length))
    return f_q


def structure_factor(lattice_points, q):
    """
    Compute the structure factor for a given scattering vector.

    Parameters:
    lattice_points (np.ndarray): Array containing the lattice points.
    q (np.ndarray): The scattering vector.

    Returns:
    complex: The structure factor for the given scattering vector.
    """
    return np.sum(np.exp(1j * np.dot(lattice_points, q)))

def simulate_diffraction(lattice_points, q_values, form_factor_func, form_factor_params):
    """
    Simulate the X-ray diffraction pattern.

    Parameters:
    lattice_points (np.ndarray): Array containing the lattice points.
    q_values (np.ndarray): Array of scattering vectors.
    form_factor_func (function): The form factor function.
    form_factor_params (tuple): Parameters to pass to the form factor function.

    Returns:
    np.ndarray: Array of intensity values corresponding to the scattering vectors.
    """
    intensities = np.zeros(q_values.shape[0])
    for i, q in enumerate(q_values):
        F_q = structure_factor(lattice_points, q)
        f_q = form_factor_func(*form_factor_params, q)
        intensities[i] = np.abs(F_q)**2 * np.abs(f_q)**2
            
    return intensities

def generate_q_values_2d(num_q, q_max):
    """
    Generate a grid of q values for the simulation in 2D.

    Parameters:
    num_q (int): Number of q values along each axis.
    q_max (float): Maximum value of q.

    Returns:
    np.ndarray: Array of shape (num_q**2, 2) containing the q values.
    """
    q_coords = np.linspace(-q_max, q_max, num_q)
    qx, qy = np.meshgrid(q_coords, q_coords)
    q_values = np.column_stack([qx.ravel(), qy.ravel()])
    return q_values

def project_to_2d_plane(q_values, q_z):
    """
    Project 3D q values to a 2D plane by keeping qz constant.

    Parameters:
    q_values (np.ndarray): Array of 3D q values.
    q_z (float): The z-component of the scattering vector to keep constant.

    Returns:
    np.ndarray: Array of 2D q values.
    """
    return np.column_stack([q_values[:, 0], q_values[:, 1], np.full(q_values.shape[0], q_z)])

def euler_rotation_matrix(phi, theta, psi):
    """
    Create a rotation matrix from Euler angles.

    Parameters:
    phi (float): Rotation angle around the z-axis.
    theta (float): Rotation angle around the y-axis.
    psi (float): Rotation angle around the x-axis.

    Returns:
    np.ndarray: Rotation matrix of shape (3, 3).
    """
    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi), np.cos(phi), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(psi), -np.sin(psi)],
        [0, np.sin(psi), np.cos(psi)]
    ])
    
    return Rz @ Ry @ Rx

def rotate_lattice(lattice_points, phi, theta, psi):
    """
    Rotate the lattice points using Euler angles.

    Parameters:
    lattice_points (np.ndarray): Array containing the lattice points.
    phi (float): Rotation angle around the z-axis.
    theta (float): Rotation angle around the y-axis.
    psi (float): Rotation angle around the x-axis.

    Returns:
    np.ndarray: Rotated lattice points.
    """
    rotation_matrix = euler_rotation_matrix(phi, theta, psi)
    rotated_points = lattice_points @ rotation_matrix.T
    return rotated_points

def plot_side_by_side(lattice_points, q_values_2d, intensities, q_max):
    """
    Plot the 3D lattice and the 2D diffraction pattern side by side.

    Parameters:
    lattice_points (np.ndarray): Array containing the lattice points.
    q_values_2d (np.ndarray): Array of 2D scattering vectors.
    intensities (np.ndarray): Array of intensity values.
    q_max (float): Maximum value of q.
    """
    fig = plt.figure(figsize=(12, 6))

    # Plot the 3D lattice
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(lattice_points[:, 0], lattice_points[:, 1], lattice_points[:, 2], c='b', marker='o')
    ax1.set_title('3D Lattice')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=90, azim=-90, roll=0)

    # Plot the 2D diffraction pattern
    ax2 = fig.add_subplot(122)
    scatter = ax2.scatter(q_values_2d[:, 0], q_values_2d[:, 1], c=intensities, cmap='viridis', marker='.',norm=colors.LogNorm())
    plt.colorbar(scatter, ax=ax2, label='Intensity')
    ax2.set_title('2D X-ray Diffraction Pattern')
    ax2.set_xlabel('Qx')
    ax2.set_ylabel('Qy')
    ax2.set_xlim([-q_max, q_max])
    ax2.set_ylim([-q_max, q_max])
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()
    

def hanning(image):
    #circular mask of radius=radius over image 
    xs=np.hanning(image.shape[0])
    ys=np.hanning(image.shape[1])
    temp=np.outer(xs,ys)
    return temp
    
    
def rec_hanning_window(image,iterations):
    if iterations==1:
        return image * window('hann', image.shape)
    else:
        return rec_hanning_window(image * window('hann', image.shape),iterations-1)




def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    
    
if __name__ == 'main':
    lattices=['SC','FCC','BCC']
    phis=np.linspace(0,90,12) # Rotation around the z-axis
    thetas=np.linspace(0,90,12) # Rotation around the y-axis
    psis=np.linspace(0,90,12) # Rotation around the x-axis
    form_factors=[form_factor_sphere,form_factor_cube,form_factor_tetrahedron]
    count=0
    dir_num=0
    save=False

    for lattice_type in lattices:
        for f in form_factors:
            for phi in phis:
                for theta in thetas:
                    for psi in psis:
                        count+=1
                        # Parameters for the lattice
                        # lattice_type = 'FCC'  # Type of lattice: 'SC', 'BCC', 'FCC'
                        num_points_x =  12     # Number of points along the x-axis
                        num_points_y = 12    # Number of points along the y-axis
                        num_points_z = 12     # Number of points along the z-axis
                        spacing = 8.0         # Distance between adjacent points

                        # Create the lattice
                        lattice_points = create_lattice(lattice_type, num_points_x, num_points_y, num_points_z, spacing)
                        
                        # Rotate lattice
                        rotated_lattice_points = rotate_lattice(lattice_points, phi, theta, psi)

                        # Parameters for the form factor
                        radius = spacing/2

                        # Parameters for the q values
                        num_q = 256  # Number of q values along each axis
                        q_max = 2.0  # Maximum value of q

                        # Generate 2D q values
                        q_values_2d = generate_q_values_2d(num_q, q_max)

                        # Project to 2D plane with a constant qz value (qz=0, projection approximation)
                        q_z = 0.0
                        q_values_3d = project_to_2d_plane(q_values_2d, q_z)

                        # Simulate the diffraction pattern with specified form factor
                        intensities = simulate_diffraction(rotated_lattice_points, q_values_3d, f, (radius,))

                        # Plot the 3D lattice and the 2D diffraction pattern side by side
                        plot_side_by_side(rotated_lattice_points, q_values_2d, intensities, q_max)

                        # CONVOLUTING THE DIFFRACTION PATTERNS WITH A FOCUSED PROBE
                        # convert to pixel image
                        intensities_image=intensities.reshape(num_q,num_q)
                        
                        if save:
                            np.savez('/mnt/micdata2/12IDC/ptychosaxs/data/diff_sim/{}/output_ideal_{:05d}.npz'.format(dir_num,count),idealDP=intensities_image)
                            print('file saved to output_ideal_{:05d}.npz'.format(count))
                       

        
