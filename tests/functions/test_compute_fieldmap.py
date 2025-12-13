from functions.compute_fieldmap import is_nifti, load_sus_dist, save_to_nifti, compute_bz
from functions.analytical_cases import analytical_sphere_external_field
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
    affine_matrix = np.eye(4)
    nifti_imagw = nib.Nifti1Image(data, affine_matrix)
    filepath = os.path.join(tmpdir, 'output_image.nii')
    nib.save(nifti_imagw, filepath)
    loaded_data, image_resolution, loaded_affine_matrix = load_sus_dist(filepath)

    assert np.array_equal(loaded_data, data), "load_sus_dist failed to retrive the image data correctly"
    assert np.array_equal(image_resolution, [1,1,1]), "load_sus_dist failed to retrive the image resolution correctly"
    assert np.array_equal(loaded_affine_matrix, affine_matrix), "load_sus_dist failed to retrive the affine matrix correctly"

def test_compute_bz_zero_susceptibility():
    zero_susceptibility = np.zeros((64,64,64))
    result = compute_bz(zero_susceptibility)

    assert np.array_equal(result, zero_susceptibility), "Field variation computation failed for zero susceptiility"

def test_compute_bz_uniform_susceptibility():
    uniform_susceptibility = np.ones((64,64,64))
    result = compute_bz(uniform_susceptibility)
    reference_point = result[0,0,0]

    assert np.allclose(result, reference_point), "Field variation computation failed for uniform susceptiility"

def test_save_to_nifti(tmpdir):
    data = np.random.rand(32, 32, 32)
    affine_matrix = np.eye(4)
    filepath = os.path.join(tmpdir, 'output_image.nii')
    save_to_nifti(data, affine_matrix, filepath)

    loaded_nii = nib.load(filepath)
    loaded_data = loaded_nii.get_fdata()
    loaded_affine_matrix = loaded_nii.affine

    assert np.allclose(data, loaded_data), "save_to_nifti failed to save the image data correctly"
    assert np.array_equal(affine_matrix, loaded_affine_matrix), "save_to_nifti failed to save the affine matrix correctly"

def test_compute_bz_single_water_voxel_even_matrix():
    """Test single water voxel in air for even-sized matrix [128,128,128]."""
    # Create all-air volume
    chi_air = 0.35  # ppm
    susceptibility = np.ones((128, 128, 128)) * chi_air

    # Set center voxel to water
    chi_water = -9.0  # ppm
    susceptibility[63, 63, 63] = chi_water

    # Compute field
    result = compute_bz(susceptibility, buffer=10)

    # Expected: field everywhere should be 1/3 * chi_air
    # (the dominant contribution comes from the uniform air background)
    expected_value = chi_air / 3.0

    assert np.isclose(result[63, 63, 63], expected_value, atol=0.01), \
        f"Field at center voxel should be {expected_value} ppm, got {result[63, 63, 63]} ppm"

    assert np.isclose(result[0, 0, 0], expected_value, atol=0.01), \
        f"Field at corner [0,0,0] should be {expected_value} ppm, got {result[0, 0, 0]} ppm"

def test_compute_bz_single_water_voxel_odd_matrix():
    """Test single water voxel in air for odd-sized matrix [129,129,129]."""
    # Create all-air volume
    chi_air = 0.35  # ppm
    susceptibility = np.ones((129, 129, 129)) * chi_air

    # Set center voxel to water
    chi_water = -9.0  # ppm
    susceptibility[64, 64, 64] = chi_water

    # Compute field
    result = compute_bz(susceptibility, buffer=10)

    # Expected: field everywhere should be 1/3 * chi_air
    # (the dominant contribution comes from the uniform air background)
    expected_value = chi_air / 3.0

    assert np.isclose(result[64, 64, 64], expected_value, atol=0.01), \
        f"Field at center voxel should be {expected_value} ppm, got {result[64, 64, 64]} ppm"

    assert np.isclose(result[0, 0, 0], expected_value, atol=0.01), \
        f"Field at corner [0,0,0] should be {expected_value} ppm, got {result[0, 0, 0]} ppm"

def test_compute_bz_single_water_voxel_neighbors_vs_analytical_even():
    """Test neighboring voxels match analytical spherical field for even matrix."""
    # Create all-air volume
    chi_air = 0.35  # ppm (external)
    chi_water = -9.0  # ppm (internal)
    susceptibility = np.ones((128, 128, 128)) * chi_air

    # Set center voxel to water (modeling as small sphere)
    center_idx = 63
    susceptibility[center_idx, center_idx, center_idx] = chi_water

    # Compute field
    result = compute_bz(susceptibility, buffer=10)

    # Effective radius for single cubic voxel
    radius = 0.6  # voxels

    # Test neighboring voxels in cardinal directions
    # Neighbors are at distance=1 from the water voxel
    neighbors = [
        (64, 63, 63, 1, 0, 0, '+x'),  # +x neighbor
        (62, 63, 63, -1, 0, 0, '-x'),  # -x neighbor
        (63, 64, 63, 0, 1, 0, '+y'),  # +y neighbor
        (63, 62, 63, 0, -1, 0, '-y'),  # -y neighbor
        (63, 63, 64, 0, 0, 1, '+z'),  # +z neighbor
        (63, 63, 62, 0, 0, -1, '-z'),  # -z neighbor
    ]

    for i, j, k, dx, dy, dz, direction in neighbors:
        # Calculate analytical expectation
        expected = analytical_sphere_external_field(dx, dy, dz, chi_water, chi_air, radius)

        # Get computed value
        computed = result[i, j, k]

        # Tolerance accounts for discretization effects
        assert np.isclose(computed, expected, atol=0.05), \
            f"Neighbor in {direction} direction [{i},{j},{k}]: expected {expected:.4f} ppm, got {computed:.4f} ppm"

def test_compute_bz_single_water_voxel_neighbors_vs_analytical_odd():
    """Test neighboring voxels match analytical spherical field for odd matrix."""
    # Create all-air volume
    chi_air = 0.35  # ppm (external)
    chi_water = -9.0  # ppm (internal)
    susceptibility = np.ones((129, 129, 129)) * chi_air

    # Set center voxel to water
    center_idx = 64
    susceptibility[center_idx, center_idx, center_idx] = chi_water

    # Compute field
    result = compute_bz(susceptibility, buffer=10)

    # Effective radius for single cubic voxel
    radius = 0.6  # voxels

    # Test neighboring voxels in cardinal directions
    # Neighbors are at distance=1 from the water voxel
    neighbors = [
        (65, 64, 64, 1, 0, 0, '+x'),  # +x neighbor
        (63, 64, 64, -1, 0, 0, '-x'),  # -x neighbor
        (64, 65, 64, 0, 1, 0, '+y'),  # +y neighbor
        (64, 63, 64, 0, -1, 0, '-y'),  # -y neighbor
        (64, 64, 65, 0, 0, 1, '+z'),  # +z neighbor
        (64, 64, 63, 0, 0, -1, '-z'),  # -z neighbor
    ]

    for i, j, k, dx, dy, dz, direction in neighbors:
        # Calculate analytical expectation
        expected = analytical_sphere_external_field(dx, dy, dz, chi_water, chi_air, radius)

        # Get computed value
        computed = result[i, j, k]

        # Tolerance accounts for discretization effects
        assert np.isclose(computed, expected, atol=0.05), \
            f"Neighbor in {direction} direction [{i},{j},{k}]: expected {expected:.4f} ppm, got {computed:.4f} ppm"
