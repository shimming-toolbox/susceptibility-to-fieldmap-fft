import numpy as np
import nibabel as nib
import argparse

def is_nifti(filepath):
    """
    Check if a file is in the NIfTI format.

    Arguments:
        filepath (str): the filepath to check

    Returns:
        bool: True if the file is NifTI, False otherwise.
    """
    if filepath[-4:] == '.nii':
        return True
    else:
        return False

def load_sus_dist(filepath):
    """
    Load the susceptibility data and the image resolution from a NiFTI file.

    Arguments:
        filepath (str): the file containing the susceptibility distribution

    Returns:
        (sus_dist, img_res): A tuple containing the susceptibility data (np.array)
                            and the image resolution (np.array)
    """    
    img = nib.load(filepath)
    sus_dist = img.get_fdata()
    header = img.header
    img_res = np.array(header.get_zooms())

    return sus_dist, img_res

def compute_bz(sus_dist, img_res=np.array([1,1,1]), buffer=1):
    """
    Compute the Bz field variation based on a susceptibility distribution
    using a Fourier-based method.
    
    """
    # dimensions needs to be a numpy.array
    dimensions = np.array(sus_dist.shape)

    # creating the k-space grid
    dim = buffer*np.array(dimensions)
    kmax = 1/(2*img_res)
    interval = 2*kmax/dim

    [kx, ky, kz] = np.meshgrid(np.arange(-kmax[0], kmax[0], interval[0]),
                                np.arange(-kmax[1], kmax[1], interval[1]),
                                np.arange(-kmax[2], kmax[2], interval[2]))

    # FFT kernel
    k2 = kx**2 + ky**2 + kz**2

    # undetermined at the center of k-space
    with np.errstate(divide='ignore', invalid='ignore'):
        kernel = np.fft.fftshift(1/3 - kz**2/k2)
        kernel[0,0,0] = 1/3

    FT_chi = np.fft.fftn(sus_dist, dim)
    Bz_fft = kernel*FT_chi

    # retrive the inital FOV
    volume_buffed = np.real(np.fft.ifftn(Bz_fft))
    volume_without_buff = volume_buffed[0:dimensions[0], 0:dimensions[1], 0:dimensions[2]]

    return volume_without_buff

def save_to_nifti(data, img_res, output_path):
    """
    Save data to NIfTI format to a specified output path.

    Arguments:
        data (np.array): the data to save
        img_res (np.array): the spatial resolution of the data
        output_path (str): the output path to save the file

    Returns:
        None
    """
    affine = np.diag(np.append(img_res,1))
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)

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

    