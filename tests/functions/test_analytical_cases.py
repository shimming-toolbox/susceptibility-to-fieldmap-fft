from functions.analytical_cases import compare_to_analytical_internal
import numpy as np
import nibabel as nib
import os
import pytest
import matplotlib
matplotlib.use("Agg")

DEFAULT_MATRIX = (128, 128, 128)
ACCEPTABLE_ATOL = 0.1

def create_boundary_exclusion_mask(matrix, image_res, radius, ring_thickness_voxels=10):
    """Create a mask that excludes a ring around the sphere boundary.

    This mask excludes regions within ±ring_thickness_voxels of the sphere
    surface to avoid Gibbs ringing artifacts in FFT-based field calculations.

    Args:
        matrix: Array dimensions [nx, ny, nz]
        image_res: Image resolution in mm [dx, dy, dz]
        radius: Sphere radius in mm
        ring_thickness_voxels: Thickness of boundary ring to exclude (in voxels)

    Returns:
        Boolean mask: True for valid comparison regions, False for excluded ring
    """
    [x, y, z] = np.meshgrid(np.linspace(-(matrix[0]-1)/2, (matrix[0]-1)/2, matrix[0]),
                            np.linspace(-(matrix[1]-1)/2, (matrix[1]-1)/2, matrix[1]),
                            np.linspace(-(matrix[2]-1)/2, (matrix[2]-1)/2, matrix[2]))

    # Distance from center in voxels
    r_voxels = np.sqrt(x**2 + y**2 + z**2)

    # Sphere radius in voxels (assuming isotropic for simplicity, use image_res[0])
    radius_voxels = radius / image_res[0]

    # Exclude ring: radius_voxels - thickness < r < radius_voxels + thickness
    inner_boundary = radius_voxels - ring_thickness_voxels
    outer_boundary = radius_voxels + ring_thickness_voxels

    # True where we want to compare (outside the exclusion ring)
    valid_mask = (r_voxels < inner_boundary) | (r_voxels > outer_boundary)

    return valid_mask

def allclose_masked(A, F, mask, rtol=1e-5, atol=1e-8):
    """Check if arrays are close using np.allclose, only in masked regions.

    Args:
        A: Actual/analytical values
        F: Forecast/calculated values
        mask: Boolean mask - only compare where True
        rtol: Relative tolerance for np.allclose
        atol: Absolute tolerance for np.allclose

    Returns:
        Boolean: True if all masked values are close
    """
    A_masked = A[mask]
    F_masked = F[mask]

    return np.allclose(A_masked, F_masked, rtol=rtol, atol=atol)

class TestCore(object):
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    # --------------compare_to_analytical_internal tests-------------- #
    @pytest.mark.unit
    def test_matrix_arg_error_cases(self):
        
        geometry_type='spherical'
        buffer=1

        # Write tests for all fail cases
        with pytest.raises(TypeError):
            compare_to_analytical_internal(geometry_type, buffer, matrix=128)
        
        with pytest.raises(ValueError):
            compare_to_analytical_internal(geometry_type, buffer, matrix=[128, 128])

        with pytest.raises(TypeError):
            compare_to_analytical_internal(geometry_type, buffer, matrix=[128.5, 128.5, 128.5])         

        with pytest.raises(ValueError):
            compare_to_analytical_internal(geometry_type, buffer, matrix=[-128, -128, -128])

    @pytest.mark.unit
    def test_imres_arg_error_cases(self):
        
        geometry_type='spherical'
        buffer=1
        matrix=[128,128,128]

        # Write tests for all fail cases
        with pytest.raises(TypeError):
            compare_to_analytical_internal(geometry_type, buffer, matrix=matrix, image_res=1)
        
        with pytest.raises(ValueError):
            compare_to_analytical_internal(geometry_type, buffer, matrix=matrix, image_res=[1, 1])

        with pytest.raises(TypeError):
            compare_to_analytical_internal(geometry_type, buffer, matrix=matrix, image_res=['1', '1', '1'])         

        with pytest.raises(ValueError):
            compare_to_analytical_internal(geometry_type, buffer,matrix= matrix, image_res=[-1, -1, -1])      

    @pytest.mark.integration
    def test_compare_analytical_spherical_default_args(self):
        
        geometry_type='spherical'
        buffer=1

        compare_to_analytical_internal(geometry_type, buffer)

    @pytest.mark.integration
    def test_compare_analytical_spherical_with_boundary_exclusion(self):
        """Test spherical field excluding Gibbs ringing at boundary."""
        geometry_type = 'spherical'
        buffer = 1
        matrix = [128, 128, 128]
        image_res = [1, 1, 1]
        radius = 15

        calculated_Bz, Bz_analytical = compare_to_analytical_internal(
            geometry_type, buffer, matrix=matrix, image_res=image_res, radius=radius
        )

        # Exclude ±10 pixel ring around sphere boundary
        exclusion_mask = create_boundary_exclusion_mask(
            matrix, image_res, radius, ring_thickness_voxels=10
        )

        # Test with allclose in valid regions only (0.1 ppm absolute tolerance)
        assert allclose_masked(Bz_analytical, calculated_Bz, exclusion_mask, atol=0.1)

    # Buffer tests
    @pytest.mark.integration
    def test_compare_analytical_spherical_default_buffer(self):
        
        geometry_type='spherical'
        buffer=1
        compare_to_analytical_internal(geometry_type, buffer=buffer)
        
    @pytest.mark.integration
    def test_compare_analytical_spherical_default_buffer_expected_matrix_shape(self):
        
        geometry_type='spherical'
        buffer=1
        calculated_Bz, Bz_analytical = compare_to_analytical_internal(geometry_type, buffer=buffer)

        assert calculated_Bz.shape == DEFAULT_MATRIX
        assert Bz_analytical.shape == DEFAULT_MATRIX
        
    @pytest.mark.integration
    def test_compare_analytical_spherical_zero_buffer(self):

        geometry_type='spherical'
        buffer=0
        compare_to_analytical_internal(geometry_type, buffer=buffer)

    @pytest.mark.integration
    def test_compare_analytical_spherical_zero_buffer_expected_matrix(self):
        
        geometry_type='spherical'
        buffer=0
        calculated_Bz, Bz_analytical = compare_to_analytical_internal(geometry_type, buffer=buffer)

        assert calculated_Bz.shape == DEFAULT_MATRIX
        assert Bz_analytical.shape == DEFAULT_MATRIX

    @pytest.mark.integration
    def test_compare_analytical_spherical_twopix_buffer(self):

        geometry_type='spherical'
        buffer=2
        calculated_Bz, Bz_analytical = compare_to_analytical_internal(geometry_type, buffer=buffer)

        assert calculated_Bz.shape == DEFAULT_MATRIX
        assert Bz_analytical.shape == DEFAULT_MATRIX

    # This next test currently gets killed from a memory error
    #@pytest.mark.integration
    @pytest.mark.xfail
    def test_compare_analytical_spherical_64px_buffer(self):
        assert False    
    #    geometry_type='spherical'
    #    buffer=64
    #    compare_to_analytical_internal(geometry_type, buffer=buffer)

    # Matrix tests
    @pytest.mark.integration
    def test_compare_analytical_spherical_alleven_matrix(self):
        
        geometry_type='spherical'
        buffer=1
        matrix=[128,128,128]
        compare_to_analytical_internal(geometry_type, buffer, matrix=matrix)

    @pytest.mark.integration
    def test_compare_analytical_spherical_allodd_matrix(self):
        
        geometry_type='spherical'
        buffer=1
        matrix=[129,129,129]
        compare_to_analytical_internal(geometry_type, buffer, matrix=matrix)

    #@pytest.mark.integration
    @pytest.mark.xfail
    def test_compare_analytical_spherical_mixed_pairity_matrix(self):
        
        geometry_type='spherical'
        buffer=1
        matrix=[128,129,128]
        compare_to_analytical_internal(geometry_type, buffer, matrix=matrix)

    # Image res tests
    @pytest.mark.integration
    def test_compare_analytical_spherical_allodd_image_res(self):
        
        geometry_type='spherical'
        buffer=1
        image_res=[1,1,1]
        compare_to_analytical_internal(geometry_type, buffer, image_res=image_res)

    @pytest.mark.integration
    def test_compare_analytical_spherical_alleven_image_res(self):
        
        geometry_type='spherical'
        buffer=1
        image_res=[2,2,2]
        compare_to_analytical_internal(geometry_type, buffer, image_res=image_res)

    @pytest.mark.integration
    def test_compare_analytical_spherical_anisotropic_image_res(self):
        
        geometry_type='spherical'
        buffer=1
        image_res=[1,1,2]
        compare_to_analytical_internal(geometry_type, buffer, image_res=image_res)

    # Radius tests
    @pytest.mark.integration
    def test_compare_analytical_spherical_default_radius(self):
        
        geometry_type='spherical'
        buffer=1
        radius=15
        compare_to_analytical_internal(geometry_type, buffer, radius=radius)

    @pytest.mark.integration
    def test_compare_analytical_spherical_small_radius(self):
        
        geometry_type='spherical'
        buffer=1
        radius=1
        compare_to_analytical_internal(geometry_type, buffer, radius=radius)

    @pytest.mark.integration
    def test_compare_analytical_spherical_large_radius(self):
        
        geometry_type='spherical'
        buffer=1
        radius=128
        compare_to_analytical_internal(geometry_type, buffer, radius=radius)

    # Chi tests
    @pytest.mark.integration
    def test_compare_analytical_spherical_default_chi(self):
        
        geometry_type='spherical'
        buffer=1
        chi=9
        compare_to_analytical_internal(geometry_type, buffer, chi=chi)

    @pytest.mark.integration
    def test_compare_analytical_spherical_negative_chi(self):
        
        geometry_type='spherical'
        buffer=1
        chi=-9
        compare_to_analytical_internal(geometry_type, buffer, chi=chi)

    @pytest.mark.integration
    def test_compare_analytical_spherical_zero_chi(self):
        
        geometry_type='spherical'
        buffer=1
        chi=0
        compare_to_analytical_internal(geometry_type, buffer, chi=chi)

    @pytest.mark.integration
    def test_compare_analytical_spherical_decimal_chi(self):
        
        geometry_type='spherical'
        buffer=1
        chi=9.5
        compare_to_analytical_internal(geometry_type, buffer, chi=chi)