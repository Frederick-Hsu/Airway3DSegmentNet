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
from torch.utils.tensorboard import SummaryWriter


# Functions ========================================================================================
def visualize_airway_tree_segment_effect(epoch_num,
                                         case_name,
                                         raw_image_cuboid,
                                         label_cuboid,
                                         segment_cuboid,
                                         save_dir,
                                         tensorboard_writer,
                                         total_count):
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

    fig = plt.figure(figsize=(20, 20))
    plt.title("{0}: airway tree segmentation at epoch{1}".format(case_name, epoch_num))
    plt.xlabel("cuboid width")
    plt.ylabel("cuboid depth")
    plt.imshow(raw_image_cuboid_front_view, cmap='gray')
    plt.contour(label_cuboid_front_view, colors='r')
    plt.contour(segment_cuboid_front_view, colors='g')
    fig_file = os.path.join(save_dir,
                            "{0}_airway_tree_segmentation_at_epoch{1}.png".format(case_name, epoch_num))
    plt.savefig(fig_file)
    # Read the figure file to convert it to numpy.array format
    img = plt.imread(fig_file)

    total_count += 1
    tensorboard_writer.add_image(tag="{0}: airway tree segmentation at epoch{1}".format(case_name, epoch_num),
                                 img_tensor=img,
                                 global_step=total_count,
                                 dataformats="HWC")
    tensorboard_writer.flush()
    plt.close(fig)


def visualize_bronchus_segment_slices(epoch_num,
                                      case_name,
                                      raw_image_cuboid,
                                      label_cuboid,
                                      segment_cuboid,
                                      save_dir,
                                      tensorboard_writer,
                                      total_count):
    depth, height, width = label_cuboid.shape

    # Extract 5 slices equally from the raw_image_cuboid
    for index in range(1, 6):
        slice_index = depth * index // 6
        image_slice = raw_image_cuboid[slice_index, :, :]
        label_slice = label_cuboid[slice_index, :, :]
        segment_slice = segment_cuboid[slice_index, :, :]
        assert len(image_slice.shape) == 2 and (image_slice.shape == label_slice.shape == segment_slice.shape)

        fig = plt.figure(figsize=(20, 20))
        plt.title("{0}: bronchus slice #{1}, overlapping label and segment".format(case_name, slice_index))
        plt.xlabel("cuboid width")
        plt.ylabel("cuboid height")
        plt.imshow(image_slice, cmap='gray')
        plt.contour(segment_slice, colors='r')
        plt.contour(label_slice, colors='g')

        fig_file = os.path.join(save_dir,
            "{0}_bronchus_segmentation_slice{1}_at_epoch{2}.png".format(case_name, slice_index, epoch_num))
        plt.savefig(fig_file)
        plt.close(fig)

        img = plt.imread(fig_file)
        total_count += 1
        tensorboard_writer.add_image(tag="{0}: bronchus segmentation at epoch{1}".format(case_name, epoch_num),
                                     img_tensor=img,
                                     global_step=total_count,
                                     dataformats="HWC")
        tensorboard_writer.flush()


#===================================================================================================
if __name__ == "__main__":
    label_cuboid_npy_file = "label_cuboid.npy"
    raw_image_cuboid_npy_file = "raw_image_cuboid.npy"
    segment_cuboid_npy_file = "predict_cuboid.npy"

    label_cuboid_npy = np.load(label_cuboid_npy_file)
    segment_cuboid_npy = np.load(segment_cuboid_npy_file)
    raw_image_cuboid_npy = np.load(raw_image_cuboid_npy_file)

    total_count = 1
    tensorboard_writer = SummaryWriter(log_dir="tensorboard")
    visualize_airway_tree_segment_effect(case_name='ATM_015_0000',
                                         raw_image_cuboid=raw_image_cuboid_npy,
                                         label_cuboid=label_cuboid_npy,
                                         segment_cuboid=segment_cuboid_npy,
                                         epoch_num=5,
                                         save_dir="./",
                                         tensorboard_writer=tensorboard_writer,
                                         total_count=total_count)

    visualize_bronchus_segment_slices(epoch_num=5,
                                      case_name='ATM_015_0000',
                                      raw_image_cuboid=raw_image_cuboid_npy,
                                      label_cuboid=label_cuboid_npy,
                                      segment_cuboid=segment_cuboid_npy,
                                      save_dir='./',
                                      tensorboard_writer=tensorboard_writer,
                                      total_count=total_count)
    tensorboard_writer.close()