from fft_simulation.fft_simulation import is_nifti, load_sus_dist, compute_bz
import numpy as np
import nibabel as nib
import os


def test_is_nifti():
    good_filepath = 'example.nii'
    wrong_filepath = 'example.txt'

    assert is_nifti(good_filepath) and is_nifti(wrong_filepath) is False, "is_nifti is incorrect"
    
def test_load_sus_dist(tmpdir):
    data = np.random.rand(32, 32, 32)
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(data, affine)
    filepath = os.path.join(tmpdir, 'output_image.nii')
    nib.save(nifti_img, filepath)
    loaded_data, img_res = load_sus_dist(filepath)

    assert np.array_equal(loaded_data, data) and np.array_equal(img_res, [1,1,1])





