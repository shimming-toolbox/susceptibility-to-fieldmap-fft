import nibabel as nib
import numpy as np

vol = nib.load('sub-amuAL_T1w_label-all.nii.gz')
#vol = nib.load('sub-unfErssm010_T1w_label-all.nii.gz')


vol_data = vol.get_fdata()

vol_data.astype(np.float64)

vol_data.fill(0)

# empty space
new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)
nib.save(new_volume, 'zeroes.nii.gz')
#nib.save(new_volume, 'sub-unfErssm010_T1w_label-all-chi.nii.gz')

# air
vol_data.fill(0.35)
new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)
nib.save(new_volume, 'air.nii.gz')

# water

vol_data.fill(-9.05)
new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)
nib.save(new_volume, 'air.nii.gz')

# Delta water

dims = np.shape(vol_data)
vol_data.fill(0)
vol_data[int(np.round(dims[0]/2)), int(np.round(dims[1]/2)), int(np.round(dims[2]/2))]=-9.05
new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)
nib.save(new_volume, 'delta.nii.gz')

# Top x half water

dims = np.shape(vol_data)
vol_data.fill(0.35)
vol_data[:int(np.round(dims[0]/2)), :, :]=-9.05
new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)

nib.save(new_volume, 'top_x_half.nii.gz')

# Bottom x half water

dims = np.shape(vol_data)
vol_data.fill(0.35)
vol_data[int(np.round(dims[0]/2)):, :, :]=-9.05
new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)

nib.save(new_volume, 'bottom_x_half.nii.gz')


# Top y half water

dims = np.shape(vol_data)
vol_data.fill(0.35)
vol_data[:, :int(np.round(dims[1]/2)), :]=-9.05
new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)

nib.save(new_volume, 'top_y_half.nii.gz')

# Bottom y half water

dims = np.shape(vol_data)
vol_data.fill(0.35)
vol_data[:, int(np.round(dims[1]/2)):, :]=-9.05
new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)

nib.save(new_volume, 'bottom_y_half.nii.gz')

# Top z half water

dims = np.shape(vol_data)
vol_data.fill(0.35)
vol_data[:, :, :int(np.round(dims[2]/2))]=-9.05
new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)

nib.save(new_volume, 'top_z_half.nii.gz')

# Bottom z half water

dims = np.shape(vol_data)
vol_data.fill(0.35)
vol_data[:, :, int(np.round(dims[2]/2)):]=-9.05
new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)

nib.save(new_volume, 'bottom_z_half.nii.gz')

# Halfbox water

dims = np.shape(vol_data)
vol_data.fill(0.35)
vol_data[int(np.round(dims[0]*1/4)):int(np.round(dims[2]*3/4)), int(np.round(dims[1]*1/4)):int(np.round(dims[2]*3/4)), int(np.round(dims[2]*1/4)):int(np.round(dims[2]*3/4))]=-9.05
new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)

nib.save(new_volume, 'halfbox.nii.gz')

# Box water
dims = np.shape(vol_data)
vol_data.fill(0.35)
vol_data[int(np.round(dims[0]*1/4)):int(np.round(dims[0]*3/4)), int(np.round(dims[1]*1/4)):int(np.round(dims[1]*3/4)), int(np.round(dims[2]*1/4)):int(np.round(dims[2]*3/4))]=-9.05
new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)

nib.save(new_volume, 'box.nii.gz')

# Mostlyfilled
dims = np.shape(vol_data)
vol_data.fill(0.35)
vol_data[int(np.round(dims[0]*0.05)):int(np.round(dims[0]*0.95)), int(np.round(dims[1]*0.05)):int(np.round(dims[1]*0.95)), int(np.round(dims[2]*0.05)):int(np.round(dims[2]*0.95))]=-9.05
new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)

nib.save(new_volume, 'mostlyfilled.nii.gz')



# Halfbox water - buffer

dims = np.shape(vol_data)
vol_data.fill(0.35)
vol_data[int(np.round(dims[0]*1/4)):int(np.round(dims[2]*3/4)), int(np.round(dims[1]*1/4)):int(np.round(dims[2]*3/4)), int(np.round(dims[2]*1/4)):int(np.round(dims[2]*3/4))]=-9.05
new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)

nib.save(new_volume, 'halfbox_buffer.nii.gz')
