import numpy as np
import nibabel as nib
import argparse

def is_nifti(filepath):
    """
    Check if the given filepath represents a NIfTI file.

    Args:
        filepath (str): The path of the file to check.

    Returns:
        bool: True if the file is a NIfTI file, False otherwise.
    """
    if filepath[-4:] == '.nii':
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

    return susceptibility_distribution, image_resolution


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
    interval = 2*kmax/new_dimensions

    [kx, ky, kz] = np.meshgrid(np.arange(-kmax[0], kmax[0], interval[0]),
                                np.arange(-kmax[1], kmax[1], interval[1]),
                                np.arange(-kmax[2], kmax[2], interval[2]))

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

def save_to_nifti(data, image_resolution, output_path):
    """
    Save data to NIfTI format to a specified output path.

    Args:
        data (np.array): the data to save
        img_res (np.array): the spatial resolution of the data
        output_path (str): the output path to save the file

    Returns:
        None
    """
    affine = np.diag(np.append(image_resolution,1))
    nifti_image = nib.Nifti1Image(data, affine)
    nib.save(nifti_image, output_path)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        dest="input_file",
                        type=str,
                        required=True,
                        help="Path to the NIfTI file input.")
    
    parser.add_argument("-o",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="Path to the NIfTI file ouput.")
    args = parser.parse_args()

    if is_nifti(args.input_file):
        sus_dist, img_res = load_sus_dist(args.input_file)
        fieldmap = compute_bz(sus_dist, img_res)
        save_to_nifti(fieldmap, img_res, args.output_file)
    else:
        print("The input file is not NIfTI.")


if __name__ == "__main__":
    main()

    