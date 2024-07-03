from fft_simulation.fft_simulation import is_nifti, load_sus_dist, save_to_nifti, compute_bz
import numpy as np
import nibabel as nib
import os


def test_is_nifti():
    good_filepath1 = 'example.nii'
    good_filepath2 = 'example.nii.gz'
    wrong_filepath = 'example.txt'

    assert is_nifti(good_filepath1), "is_nifti failed for a .nii file"
    assert is_nifti(good_filepath2), "is_nifti failed for a .nii.gz file"
    assert is_nifti(wrong_filepath) is False, "is_nifti failed for a wrong filepath"
    
def test_load_sus_dist(tmpdir):
    data = np.random.rand(32, 32, 32)
    affine = np.eye(4)
    nifti_imagw = nib.Nifti1Image(data, affine)
    filepath = os.path.join(tmpdir, 'output_image.nii')
    nib.save(nifti_imagw, filepath)
    loaded_data, image_resolution = load_sus_dist(filepath)

    assert np.array_equal(loaded_data, data), "load_sus_dist failed to retrive the image data correctly"
    assert np.array_equal(image_resolution, [1,1,1]), "load_sus_dist failed to retrive the image resolution correctly"

def test_compute_bz_zero_susceptibility():
    zero_susceptibility = np.zeros((64,64,64))
    result = compute_bz(zero_susceptibility)

    assert np.array_equal(result, zero_susceptibility), "Field variation computation failed for zero susceptiility"

def test_compute_bz_uniform_susceptibility():
    uniform_susceptibility = np.ones((64,64,64))
    result = compute_bz(uniform_susceptibility)
    reference_point = result[0,0,0]

    assert np.all(result == reference_point), "Field variation computation failed for uniform susceptiility"

def test_save_to_nifti(tmpdir):
    data = np.random.rand(32, 32, 32)
    image_resolution = np.array([1,1,1])
    filepath = os.path.join(tmpdir, 'output_image.nii')
    save_to_nifti(data, image_resolution, filepath)

    loaded_nii = nib.load(filepath)
    header = loaded_nii.header
    loaded_data = loaded_nii.get_fdata()
    loaded_image_resolution = header.get_zooms()

    assert np.array_equal(data, loaded_data), "save_to_nifti failed to save the image data correctly"
    assert np.array_equal(image_resolution, loaded_image_resolution), "save_to_nifti failed to save the image resolution correctly"




