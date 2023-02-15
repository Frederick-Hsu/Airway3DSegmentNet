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
from metrics_calculation import save_airway_tree_3dmodel_annotated_with_3colors, \
                                save_airway_tree_3dmodel_annotated_with_2colors


# Functions ========================================================================================
def visualize_airway_tree_segment_effect(epoch_num,
                                         phase,
                                         case_name,
                                         label_cuboid,
                                         segment_cuboid,
                                         origin,
                                         spacing,
                                         save_dir,
                                         tensorboard_writer):
    assert (label_cuboid.shape == segment_cuboid.shape), "The 3 cuboids must have the same shape."

    niigz_file_name = "{0}/{1}_airway_tree_with_3colors_at_{2}_epoch{3}.nii.gz".format(save_dir,
                                                                                       case_name,
                                                                                       phase,
                                                                                       epoch_num)
    metrics = save_airway_tree_3dmodel_annotated_with_3colors(label_cuboid,
                                                              segment_cuboid,
                                                              origin,
                                                              spacing,
                                                              niigz_file_name)
    FPR, TPR, FNR, DSC = metrics
    tensorboard_writer.add_scalar(tag="{0}: False Positive Rate at {1} phase".format(case_name, phase),
                                  scalar_value=FPR,
                                  global_step=epoch_num)
    tensorboard_writer.add_scalar(tag="{0}: True Positive Rate at {1} phase".format(case_name, phase),
                                  scalar_value=TPR,
                                  global_step=epoch_num)
    tensorboard_writer.add_scalar(tag="{0}: False Negative Rate at {1} phase".format(case_name, phase),
                                  scalar_value=FPR,
                                  global_step=epoch_num)
    tensorboard_writer.add_scalar(tag="{0}: DSC at {1} phase".format(case_name, phase),
                                  scalar_value=DSC,
                                  global_step=epoch_num)

    niigz_file_name = "{0}/{1}_airway_tree_with_2colors_at_{2}_epoch{3}.nii.gz".format(save_dir,
                                                                                       case_name,
                                                                                       phase,
                                                                                       epoch_num)
    save_airway_tree_3dmodel_annotated_with_2colors(label_cuboid, segment_cuboid, origin, spacing, niigz_file_name)




def visualize_bronchus_segment_slices(epoch_num,
                                      phase,
                                      case_name,
                                      raw_image_cuboid,
                                      label_cuboid,
                                      segment_cuboid,
                                      save_dir,
                                      tensorboard_writer):
    depth, height, width = label_cuboid.shape

    total_count = 0
    # Extract 5 slices equally from the raw_image_cuboid
    for index in range(1, 6):
        slice_index = depth * index // 6
        image_slice = raw_image_cuboid[slice_index, :, :]
        label_slice = label_cuboid[slice_index, :, :]
        segment_slice = segment_cuboid[slice_index, :, :]
        assert len(image_slice.shape) == 2 and (image_slice.shape == label_slice.shape == segment_slice.shape)

        fig = plt.figure()
        plt.title("{0}: bronchus slice #{1}, overlapping label and segment".format(case_name, slice_index))
        plt.xlabel("cuboid width")
        plt.ylabel("cuboid height")
        plt.imshow(image_slice, cmap='gray')
        plt.contour(segment_slice, colors='r')  # red marks the segmentation
        plt.contour(label_slice, colors='g')    # green marks the ground-truth, green <---> ground-truth

        fig_file = os.path.join(save_dir,
            "{0}_bronchus_segmentation_slice{1}_at_{2}_epoch{3}.png".format(case_name, slice_index, phase, epoch_num))
        plt.savefig(fig_file)
        plt.close(fig)

        img = plt.imread(fig_file)
        total_count += 1
        tensorboard_writer.add_image(tag="{0}: bronchus segmentation at {1} epoch{2}".format(case_name, phase, epoch_num),
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

    tensorboard_writer = SummaryWriter(log_dir="tensorboard")
    visualize_airway_tree_segment_effect(case_name='ATM_015_0000',
                                         raw_image_cuboid=raw_image_cuboid_npy,
                                         label_cuboid=label_cuboid_npy,
                                         segment_cuboid=segment_cuboid_npy,
                                         epoch_num=5,
                                         phase='validate',
                                         save_dir="./",
                                         tensorboard_writer=tensorboard_writer)

    visualize_bronchus_segment_slices(epoch_num=5,
                                      phase='train',
                                      case_name='ATM_015_0000',
                                      raw_image_cuboid=raw_image_cuboid_npy,
                                      label_cuboid=label_cuboid_npy,
                                      segment_cuboid=segment_cuboid_npy,
                                      save_dir='./',
                                      tensorboard_writer=tensorboard_writer)
    tensorboard_writer.close()
