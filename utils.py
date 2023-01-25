#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File  : utils.py
#
#


import pickle
import sys
import SimpleITK as sitk
import numpy as np


# Functions ========================================================================================
def save_pickle(data_dict, filename):
    with open(filename, "wb") as fh:
        pickle.dump(data_dict, fh, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, "rb") as fh:
        return pickle.load(fh)


def load_CT_scan_3D_image(niigz_file_name):
    itkimage = sitk.ReadImage(niigz_file_name)
    numpyImages = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImages, numpyOrigin, numpySpacing


# Classes ==========================================================================================
class Logger:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Main logics ======================================================================================
if __name__ == "__main__":
    pass