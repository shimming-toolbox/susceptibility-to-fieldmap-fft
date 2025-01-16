import numpy as np
import nibabel as nib  


# Create a np array with 400x400x400 voxels of 0.35

data = np.ones((400,400,400))*0.35

# Make half of the voxels equal to -9

data[0:200,0:200,0:400] = -9

# Make a wall of 50 voxels in the other half equal to -4

data[250:300,250:300,0:50] = -4

# Save the data to a nifti file

img = nib.Nifti1Image(data, np.eye(4))
nib.save(img, 'data.nii.gz')

