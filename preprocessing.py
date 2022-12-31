#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 
# File : preprocessing.py
#
#


import numpy as np
import logging
import functools


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Functions ========================================================================================
# @functools.lru_cache(1)
def crop_CT_3D_image(one_CT_3D_image, crop_cube_size, stride):
    r'''
    Crop a CT 3D image into many cubes, accroding to the `crop_cube_size` volumetric unit.

    Parameters
    ----------
    one_CT_3D_image : numpy.ndarray([depth, height, width]), or 
                      3D matrix, 
                      torch.tensor([depth, height, width])
        Input a CT 3D image, whose shape should be (Depth x Height x Width)
        
    crop_cube_size : int, or a 3-elem tuple
        The cube_size unit you want to crop the CT 3D image into many cubes
        
    stride : int, or 3-elem tuple
        the stride you want to move forward along z-axis, y-axis and x-axis respectively.

    Returns
    -------
    cropped_cube_list
    '''
    assert isinstance(crop_cube_size, (int, tuple)), "Error: the crop_cube_size must be 3-elem tuple."
    if isinstance(crop_cube_size, int):
        crop_cube_size = np.array([crop_cube_size, crop_cube_size, crop_cube_size])
    else:
        assert len(crop_cube_size) == 3
    
    crop_cube_size = (min(crop_cube_size[0], one_CT_3D_image.shape[0]),
                      min(crop_cube_size[1], one_CT_3D_image.shape[1]),
                      min(crop_cube_size[2], one_CT_3D_image.shape[2]))
    
    assert isinstance(stride, (int, tuple))
    if isinstance(stride, int):
        stride = np.array([stride, stride, stride])
    else:
        assert len(stride) == 3
    
    img_shape = one_CT_3D_image.shape
    total = len(np.arange(0, img_shape[0], stride[0])) * \
            len(np.arange(0, img_shape[1], stride[1])) * \
            len(np.arange(0, img_shape[2], stride[2]))
    
    cropped_cube_list = []
    
    count = 0
    for i in np.arange(0, img_shape[0], stride[0]):
        for j in np.arange(0, img_shape[1], stride[1]):
            for k in np.arange(0, img_shape[2], stride[2]):
                logger.warning("Progress: crop one CT-3D-image {0}/{1} ".format(count+1, total))
                
                if i + crop_cube_size[0] <= img_shape[0]:
                    x_start_input   = i
                    x_end_input     = i + crop_cube_size[0]
                    x_start_output  = i
                    x_end_output    = i + stride[0]
                else:
                    x_start_input   = img_shape[0] - crop_cube_size[0]
                    x_end_input     = img_shape[0]
                    x_start_output  = i
                    x_end_output    = img_shape[0]
                
                if j + crop_cube_size[1] <= img_shape[1]:
                    y_start_input   = j
                    y_end_input     = j + crop_cube_size[1]
                    y_start_output  = j
                    y_end_output    = j + stride[1]
                else:
                    y_start_input   = img_shape[1] - crop_cube_size[1]
                    y_end_input     = img_shape[1]
                    y_start_output  = j
                    y_end_output    = img_shape[1]
                
                if k + crop_cube_size[2] <= img_shape[2]:
                    z_start_input   = k
                    z_end_input     = k + crop_cube_size[2]
                    z_start_output  = k
                    z_end_output    = k + stride[2]
                else:
                    z_start_input   = img_shape[2] - crop_cube_size[2]
                    z_end_input     = img_shape[2]
                    z_start_output  = k
                    z_end_output    = img_shape[2]
                
                crop_cube = one_CT_3D_image[x_start_input:x_end_input,
                                            y_start_input:y_end_input,
                                            z_start_input:z_end_input]
                cropped_cube_list.append(np.array(crop_cube, dtype=float))
                
                count += 1
    
    return cropped_cube_list


# Classes ==========================================================================================



# Main logics ======================================================================================
if __name__ == "__main__":
    import os
    
    current_trainset_path = "./ATM22_train"
    current_labelset_path = "./ATM22_label"
    current_validateset_path = "./ATM22_validate"
    if not os.path.exists(current_trainset_path):
        os.mkdir(current_trainset_path)
    if not os.path.exists(current_labelset_path):
        os.mkdir(current_labelset_path)
    if not os.path.exists(current_validateset_path):
        os.mkdir(current_validateset_path)
    
    # Select one raw CT image file from the train set
    raw_CT_image_path = "../Dataset/ATM22/imagesTr/ATM_010_0000.nii.gz"
    raw_CT_label_path = "../Dataset/ATM22/labelsTr/ATM_010_0000.nii.gz"
    
    from skimage import io
    one_CT_3D_image = io.imread(raw_CT_image_path, plugin='simpleitk')
    # one_CT_3D_label = io.imread(raw_CT_label_path, plugin='simpleitk')
    
    crop_cube_size = (256, 256, 256)
    stride = (128, 128, 128)
    
    cropped_cube_image_list = crop_CT_3D_image(one_CT_3D_image, crop_cube_size, stride)
    # cropped_cube_label_list = crop_CT_3D_image(one_CT_3D_label, crop_cube_size, stride)
    
    