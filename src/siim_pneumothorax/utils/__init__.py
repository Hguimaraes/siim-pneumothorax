from siim_pneumothorax.utils.rle import run_length_decode
from siim_pneumothorax.utils.rle import run_length_encode
from siim_pneumothorax.utils.parse_dicom import parse_dicom

__all__ = ['run_length_decode', 'run_length_encode', 'parse_dicom', 'ConfigObject']

class ConfigObject:
    """Transform a dictionary in a object"""
    def __init__(self, **entries):
        self.__dict__.update(entries)