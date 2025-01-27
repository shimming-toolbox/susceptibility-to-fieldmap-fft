from functions.analytical_cases import compare_to_analytical_internal
import numpy as np
import nibabel as nib
import os
import pytest
import copy
import matplotlib
matplotlib.use("Agg")

DEFAULT_MATRIX = (128, 128, 128)
ACCEPTABLE_ATOL = 0.1

def smape(A, F):
    A = copy.deepcopy(A)
    F = copy.deepcopy(F)
    A = A.reshape(-1)
    F = F.reshape(-1)

    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

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

    #@pytest.mark.single
    @pytest.mark.xfail
    def test_compare_analytical_spherical_default_outputs_close(self):
        
        geometry_type='spherical'
        buffer=1

        calculated_Bz, Bz_analytical = compare_to_analytical_internal(geometry_type, buffer)

        assert smape(Bz_analytical,calculated_Bz) < 10

    # Buffer tests
    @pytest.mark.integration
    def test_compare_analytical_spherical_default_buffer(self):
        
        geometry_type='spherical'
        buffer=1
        compare_to_analytical_internal(geometry_type, buffer=buffer)
        
    #@pytest.mark.integration
    @pytest.mark.xfail
    def test_compare_analytical_spherical_default_buffer_expected_matrix_shape(self):
        
        geometry_type='spherical'
        buffer=1
        calculated_Bz, Bz_analytical = compare_to_analytical_internal(geometry_type, buffer=buffer)

        assert calculated_Bz.shape == DEFAULT_MATRIX
        assert Bz_analytical.shape == DEFAULT_MATRIX
        
    #@pytest.mark.integration
    @pytest.mark.xfail
    def test_compare_analytical_spherical_zero_buffer(self):
        
        geometry_type='spherical'
        buffer=0
        compare_to_analytical_internal(geometry_type, buffer=buffer)

    #@pytest.mark.integration
    @pytest.mark.xfail
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
    @pytest.mark.integration
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
    #@pytest.mark.integration
    @pytest.mark.xfail
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