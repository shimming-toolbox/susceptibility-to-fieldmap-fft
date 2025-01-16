from functions.analytical_cases import compare_to_analytical_internal
import numpy as np
import nibabel as nib
import os
import pytest


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


