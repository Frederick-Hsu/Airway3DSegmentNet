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

smooth = 1.0

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

def dice_coefficient_np(predict_data, groundtruth_data):
    r'''
    Calculate the dice coefficient in the numpy type.

    :param predict_data:
    :param groundtruth_data:
    both the 2 parameters must be numpy type

    :return the dice coefficient value, in the numpy type
    '''
    groundtruth_data_flatten = groundtruth_data.flatten()
    predict_data_flatten = predict_data.flatten()
    intersection = np.sum(predict_data_flatten * groundtruth_data_flatten)
    return 2.0 * (intersection + smooth) / (np.sum(predict_data_flatten) +
                                            np.sum(groundtruth_data_flatten) +
                                            smooth)


def positive_predictive_value_np(predict_data, groundtruth_data):
    predict_data_flatten = predict_data.flatten()
    groundtruth_data_flatten = groundtruth_data.flatten()
    intersection = np.sum(predict_data_flatten * groundtruth_data_flatten)
    return (intersection + smooth) / (np.sum(predict_data_flatten) + smooth)


def sensitivity_np(predict, groundtruth):
    predict_flatten = predict.flatten()
    groundtruth_flatten = groundtruth.flatten()
    intersection = np.sum(predict_flatten * groundtruth_flatten)
    return (intersection + smooth) / (np.sum(groundtruth_flatten) + smooth)


def accuracy_np(predict, groundtruth):
    predict_flatten = predict.flatten()
    groundtruth_flatten = groundtruth.flatten()
    intersection = np.sum(predict_flatten == groundtruth_flatten)
    return intersection / (len(groundtruth_flatten) + smooth)


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