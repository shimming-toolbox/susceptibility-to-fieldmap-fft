import nibabel as nib
import numpy as np
import argparse
from pathlib import Path
import os

def main(bids_subject_dir):

    bids_subject_dir = Path(bids_subject_dir)
    subject = str(bids_subject_dir.stem)
    merged_labels_path = bids_subject_dir / ".. " / 'derivatives' / 'labels' / subject / 'anat' (subject + '_T1w_label-all.nii.gz')
    

    vol = nib.load(merged_labels_path.resolve())

    vol_data = vol.get_fdata()

    vol_data.astype(np.float64)

    vol_data[vol_data==0] = 0.35
    vol_data[vol_data==1] = -9.05
    vol_data[vol_data==2] = -2
    vol_data[vol_data==3] = -2
    vol_data[vol_data==4] = -4.2
    vol_data[vol_data==5] = -4.2
    vol_data[vol_data==6] = -4.2
    vol_data[vol_data==56] = -9.04
    vol_data[vol_data==60] = -9.05
    vol_data[vol_data==91] = -11
    vol_data[vol_data==92] = -11
    vol_data[vol_data==93] = -9.055
    vol_data[vol_data==100] = -9.055

    new_volume = nib.Nifti1Image(vol_data, vol.affine, vol.header,  dtype=np.float64)

    # Check if the directory  bids_subject_dir / ".. " / 'derivatives' / subject / exists and if no, create it

    if not os.path.exists(bids_subject_dir / ".. " / 'derivatives' / subject ):
        os.makedirs(bids_subject_dir / ".. " / 'derivatives' / subject )

    # Check if the directory  bids_subject_dir / ".. " / 'derivatives' / subject / 'anat' exists and if no, create it

    if not os.path.exists(bids_subject_dir / ".. " / 'derivatives' / subject / 'anat' ):
        os.makedirs(bids_subject_dir / ".. " / 'derivatives' / subject / 'anat' )
    
    chi_file = bids_subject_dir / ".. " / 'derivatives' / subject / 'anat' (subject + '_T1w-chi.nii.gz')

    nib.save(new_volume, chi_file)

if __name__ == "__main__":
    
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process subject directory path and other arguments.")
    
    # Add the -s argument to the parser
    parser.add_argument("-s", "--bids_subject_dir", required=True, help="Path to the subject directory in BIDS format")

    # Parse the arguments
    args = parser.parse_args()
    main(args.bids_subject_dir)
