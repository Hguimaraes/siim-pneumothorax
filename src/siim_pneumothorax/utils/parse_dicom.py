import pydicom
import numpy as np

def parse_dicom(fn, rles_codes=None, is_training=True):
    dcm_info = pydicom.dcmread(fn)

    # Create a dictionary with pacient information (Except name)
    dcm_data = {
        'fn': fn,
        'user_id': dcm_info.PatientID,
        'user_age': int(dcm_info.PatientAge),
        'user_sex': dcm_info.PatientSex,
        'pixel_spacing': dcm_info.PixelSpacing,
        'id': dcm_info.SOPInstanceUID
    }

    # Get mask annotation
    if is_training:
        match = rles_codes[rles_codes['ImageId']==dcm_info.SOPInstanceUID]

        # get meaningful information (for train set)
        if len(match) == 0:
            dcm_data['EncodedPixels'] = np.nan
            dcm_data['has_pneumothorax'] = np.nan
        else:
            dcm_data['EncodedPixels'] = match['EncodedPixels'].values
            dcm_data['has_pneumothorax'] = np.max(match['has_pneumothorax'].values)

    return dcm_data