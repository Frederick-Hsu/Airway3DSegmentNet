#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : Visualize the 3D airway tree
#
#


from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import os


# Functions ========================================================================================
def visualize_segment_effect(case_name,
                             raw_image_cuboid,
                             label_cuboid,
                             segment_cuboid,
                             epoch_num):
    assert (raw_image_cuboid.shape == label_cuboid.shape == segment_cuboid.shape), \
        "The 3 cuboids must have the same shape."

    depth, height, width = label_cuboid.shape

    segment_cuboid_front_view = np.sum(segment_cuboid, axis=1)
    label_cuboid_front_view = np.sum(label_cuboid, axis=1)
    raw_image_cuboid_front_view = raw_image_cuboid[:, height * 3 // 4, :]

    # Flip Up/Down the cuboid
    segment_cuboid_front_view = np.flipud(segment_cuboid_front_view)
    label_cuboid_front_view = np.flipud(label_cuboid_front_view)
    raw_image_cuboid_front_view = np.flipud(raw_image_cuboid_front_view)

    plt.figure(figsize=(height//10, width//10))
    plt.title("{0}: airway tree segmentation at epoch{1}".format(case_name, epoch_num))
    plt.imshow(raw_image_cuboid_front_view, cmap='gray')
    plt.contour(label_cuboid_front_view, colors='r')
    plt.contour(segment_cuboid_front_view, colors='g')



#===================================================================================================
if __name__ == "__main__":
    label_cuboid_npy_file = "label_cuboid.npy"
    raw_image_cuboid_npy_file = "raw_image_cuboid.npy"
    segment_cuboid_npy_file = "predict_cuboid.npy"

    label_cuboid_npy = np.load(label_cuboid_npy_file)
    segment_cuboid_npy = np.load(segment_cuboid_npy_file)
    raw_image_cuboid_npy = np.load(segment_cuboid_npy_file)

    visualize_segment_effect(case_name='ATM_015_0000',
                             raw_image_cuboid=raw_image_cuboid_npy,
                             label_cuboid=label_cuboid_npy,
                             segment_cuboid=segment_cuboid_npy,
                             epoch_num=5)