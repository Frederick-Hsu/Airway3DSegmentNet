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


def save_CT_scan_3D_image(image, origin, spacing, niigz_file_name):
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))

    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, niigz_file_name, True)


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

def combine_total(output, stride, cubesize):
    r'''
    Combine all sub-volume cubes together without average overlapping areas

    :param output: list of all coordinates and voxels of sub-volumes
    :param stride:
    :param cubesize:
    '''
    # the output is from the curr_gt_info
    # curr_gt_info's structure:  [curr_gt_data,  curr_splitID, curr_num_DHW, curr_shape, curr_origin, curr_spacing]
    gt_info = output[0]
    curr_shape = gt_info[3]
    curr_origin = gt_info[4]
    curr_spacing = gt_info[5]
    curr_num_DHW = gt_info[2]

    num_Depth, num_Height, num_Width = curr_num_DHW[0], gt_info[1], gt_info[2]
    [depth, height, width] = curr_shape

    if type(cubesize) is not list:
        cubesize = [cubesize, cubesize, cubesize]

    splits = {}
    for index in range(len(output)):
        curr_gt_info = output[index]
        curr_gt_data = curr_gt_info[0]
        curr_splitID = int(curr_gt_info[1])
        splits[curr_splitID] = curr_gt_data

    Cuboid = -100000 * np.ones((depth, height, width), dtype=np.float32)
    cnt = 0
    for index_depth in range(num_Depth + 1):
        for index_height in range(num_Height + 1):
            for index_width in range(num_Width + 1):
                depth_start_index   = index_depth  * stride[0]
                depth_end_index     = index_depth  * stride[0] + cubesize[0]
                height_start_index  = index_height * stride[1]
                height_end_index    = index_height * stride[1] + cubesize[1]
                width_start_index   = index_width  * stride[2]
                width_end_index     = index_width  * stride[2] + cubesize[2]

                if depth_end_index > depth:
                    depth_start_index   = depth - cubesize[0]
                    depth_end_index     = depth
                if height_end_index > height:
                    height_start_index  = height - cubesize[1]
                    height_end_index    = height
                if width_end_index > width:
                    width_start_index   = width - cubesize[2]
                    width_end_index     = width

                split = splits[cnt]
                Cuboid[ depth_start_index:depth_end_index,
                       height_start_index:height_end_index,
                        width_start_index:width_end_index]   = split
                cnt += 1

    return Cuboid, curr_origin, curr_spacing

def combine_total_avg(output, stride, cubesize):
    r'''
    Combine all sub-volume cubes together, and average overlapping areas of prediction
    '''
    pred_info = output[0]
    curr_shape = pred_info[3]
    curr_origin = pred_info[4]
    curr_spacing = pred_info[5]

    curr_num_DHW = pred_info[2]
    num_Depth, num_Height, num_Width = curr_num_DHW[0], curr_num_DHW[1], curr_num_DHW[2]
    [depth, height, width] = curr_shape

    if type(cubesize) is not list:
        cubesize = [cubesize, cubesize, cubesize]

    splits = {}
    for index in range(len(output)):
        curr_pred_info = output[index]
        curr_predict_data = curr_pred_info[0]
        curr_splitID = int(curr_pred_info[1])
        if not curr_splitID in splits.keys():
            splits[curr_splitID] = curr_predict_data
        else:
            continue

    Cuboid = np.zeros((depth, height, width), dtype=np.float32)
    count_matrix = np.zeros((depth, height, width), dtype=np.float32)

    cnt = 0
    for index_depth in range(num_Depth + 1):
        for index_height in range(num_Height + 1):
            for index_width in range(num_Width + 1):
                depth_start_index   = index_depth  * stride[0]
                depth_end_index     = index_depth  * stride[0] + cubesize[0]
                height_start_index  = index_height * stride[1]
                height_end_index    = index_height * stride[1] + cubesize[1]
                width_start_index   = index_width  * stride[2]
                width_end_index     = index_width  * stride[2] + cubesize[2]
                if depth_end_index > depth:
                    depth_start_index  = depth - cubesize[0]
                    depth_end_index    = depth
                if height_end_index > height:
                    height_start_index = height - cubesize[1]
                    height_end_index   = height
                if width_end_index > width:
                    width_start_index = width - cubesize[2]
                    width_end_index   = width

                split = splits[cnt]
                Cuboid[ depth_start_index:depth_end_index,
                       height_start_index:height_end_index,
                        width_start_index:width_end_index] += split
                count_matrix[ depth_start_index:depth_end_index,
                             height_start_index:height_end_index,
                              width_start_index:width_end_index] += 1
                cnt += 1

    Cuboid = Cuboid / count_matrix
    return Cuboid, curr_origin, curr_spacing

def normal_min_max(image_np):
    min = np.amin(image_np)
    max = np.amax(image_np)
    norm_array = (image_np - min) / (max - min)
    return norm_array


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