from fft_simulation.fft_simulation import is_nifti, load_sus_dist, save_to_nifti, compute_bz
import numpy as np
import nibabel as nib
import os


def test_is_nifti():
    good_filepath = 'example.nii'
    wrong_filepath = 'example.txt'

    assert is_nifti(good_filepath)
    assert is_nifti(wrong_filepath) is False
    
def test_load_sus_dist(tmpdir):
    data = np.random.rand(32, 32, 32)
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(data, affine)
    filepath = os.path.join(tmpdir, 'output_image.nii')
    nib.save(nifti_img, filepath)
    loaded_data, img_res = load_sus_dist(filepath)

    assert np.array_equal(loaded_data, data) 
    assert np.array_equal(img_res, [1,1,1])

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
    img_res = np.array([1,1,1])
    filepath = os.path.join(tmpdir, 'output_image.nii')
    save_to_nifti(data, img_res, filepath)

    loaded_nii = nib.load(filepath)
    header = loaded_nii.header
    loaded_data = loaded_nii.get_fdata()
    loaded_img_res = header.get_zooms()

    assert np.array_equal(data, loaded_data)
    assert np.array_equal(img_res, loaded_img_res)




