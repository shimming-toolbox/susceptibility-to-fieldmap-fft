import numpy as np
import nibabel as nib
import click
from time import perf_counter

def is_nifti(filepath):
    """
    Check if the given filepath represents a NIfTI file.

    Args:
        filepath (str): The path of the file to check.

    Returns:
        bool: True if the file is a NIfTI file, False otherwise.
    """
    if filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        return True
    else:
        return False

def load_sus_dist(filepath):
    """
    Load the susceptibility distribution from a given file.

    Args:
        filepath (str): The path to the file containing the susceptibility distribution.

    Returns:
        tuple: A tuple containing the loaded susceptibility distribution as a numpy array 
            and the image resolution as a numpy array.
    """

    image = nib.load(filepath)
    susceptibility_distribution = image.get_fdata()
    header = image.header 
    image_resolution = np.array(header.get_zooms())
    affine_matrix = image.affine

    return susceptibility_distribution, image_resolution, affine_matrix


def compute_bz(susceptibility_distribution, image_resolution=np.array([1,1,1]), buffer=1):
    """
    Compute the Bz field variation in ppm based on a susceptibility distribution
    using a Fourier-based method.

    Args:
        susceptibility_distribution (numpy.ndarray): The 3D array representing the susceptibility distribution.

        image_resolution (numpy.ndarray, optional): The resolution of the image in each dimension. Defaults to [1, 1, 1].

        buffer (int, optional): The buffer size for the k-space grid. Defaults to 1.

    Returns:
        volume_without_buffer (numpy.ndarray): The computed magnetic field Bz in ppm.

    """

    # dimensions needs to be a numpy.array
    dimensions = np.array(susceptibility_distribution.shape)

    # creating the k-space grid with the buffer
    new_dimensions = buffer*np.array(dimensions)
    kmax = 1/(2*image_resolution)

    [kx, ky, kz] = np.meshgrid(np.linspace(-kmax[0], kmax[0], new_dimensions[0]),
                                np.linspace(-kmax[1], kmax[1], new_dimensions[1]),
                                np.linspace(-kmax[2], kmax[2], new_dimensions[2]), indexing='ij')

    # FFT procedure
    # undetermined at the center of k-space
    k2 = kx**2 + ky**2 + kz**2

    with np.errstate(divide='ignore', invalid='ignore'):
        kernel = np.fft.fftshift(1/3 - kz**2/k2)
        kernel[0,0,0] = 1/3

    FFT_chi = np.fft.fftn(susceptibility_distribution, new_dimensions)
    FFT_chi[0,0,0] = FFT_chi[0,0,0] + np.prod(new_dimensions)*susceptibility_distribution[0,0,0]
    Bz_fft = kernel*FFT_chi

    # retrive the inital FOV
    volume_with_buffer = np.real(np.fft.ifftn(Bz_fft))
    volume_without_buffer = volume_with_buffer[0:dimensions[0], 0:dimensions[1], 0:dimensions[2]]

    return volume_without_buffer

def save_to_nifti(data, affine_matrix, output_path):
    """
    Save data to NIfTI format to a specified output path.

    Args:
        data (np.array): the data to save
        affine_matrix (np.array): the affine matrix of the data
        output_path (str): the output path to save the file

    Returns:
        None
    """
    nifti_image = nib.Nifti1Image(data, affine_matrix)
    nib.save(nifti_image, output_path)



@click.command(help="Compute the magnetic field variation in ppm from a susceptibility distribution in NIfTI format.")
@click.option('-i','--input','input_file', type=click.Path(exists=True), required=True,
              help="Input susceptibility distribution, supported extensions: .nii, .nii.gz")
@click.option('-o', '--output', 'output_file', type=click.Path(), default='fieldmap.nii.gz',
              help="Output fieldmap, supported extensions: .nii, .nii.gz")
def compute_fieldmap(input_file, output_file):
    """
    Main procedure for performing the simulation.

    Args:
        input_file (str): Path to the susceptibility distribution in NIfTI format.
        output_file (str): Path for the computed fieldmap in NIfTI format.

    Returns:
        None
    """
    if is_nifti(input_file):
        start_time = perf_counter()
        print('Start')
        susceptibility_distribution, image_resolution, affine_matrix = load_sus_dist(input_file)
        print('Susceptibility distribution loaded')
        fieldmap = compute_bz(susceptibility_distribution, image_resolution)
        print('Fieldmap simulated')
        save_to_nifti(fieldmap, affine_matrix, output_file)
        print('Saving to NIfTI format')
        end_time = perf_counter()
        print(f'End. Runtime: {end_time-start_time:.2f} seconds')
    else:
        print("The input file must be NIfTI.")

    