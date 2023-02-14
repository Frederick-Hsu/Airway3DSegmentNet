#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File  : metrics_calculation.py
#
#


import numpy as np

from utils import save_CT_scan_3D_image, load_CT_scan_3D_image

# Functions ========================================================================================
def compute_fusioned_airway_tree_3dmodel(label_npy, predict_npy):
    assert label_npy.shape == predict_npy.shape

    depth, height, width = label_npy.shape
    airway_tree_3d_model = np.zeros((depth, height, width))

    false_positive_count = 0
    false_negative_count = 0
    true_positive_count = 0
    true_negative_count = 0

    for depth_index in range(depth):
        for height_index in range(height):
            for width_index in range(width):
                gt_voxel = label_npy[depth_index, height_index, width_index]
                pred_voxel = predict_npy[depth_index, height_index, width_index]

                if pred_voxel == 1 and gt_voxel == 0:  # False Positive
                    false_positive_count += 1
                    airway_tree_3d_model[depth_index, height_index, width_index] = 2
                elif pred_voxel == 1 and gt_voxel == 1:  # True Positive
                    true_positive_count += 1
                    airway_tree_3d_model[depth_index, height_index, width_index] = 3
                elif pred_voxel == 0 and gt_voxel == 0:  # True Negative
                    true_negative_count += 1
                    airway_tree_3d_model[depth_index, height_index, width_index] = 0
                elif pred_voxel == 0 and gt_voxel == 1:  # False Negative
                    false_negative_count += 1
                    airway_tree_3d_model[depth_index, height_index, width_index] = 1

    return airway_tree_3d_model, [false_positive_count,
                                  false_negative_count,
                                  true_positive_count,
                                  true_negative_count]

def save_airway_tree_3dmodel_annotated_with_3colors(groundtruth_npy, predict_npy, origin, spacing, niigz_file_name):
    airway_tree, metrics = compute_fusioned_airway_tree_3dmodel(groundtruth_npy, predict_npy)

    # Note that the colors
    # 0               True Negative voxel                  Black                    the black background
    # 1               False Negative voxel                 Red
    # 2               False Positive voxel                 Green
    # 3               True Positive voxel                  Blue
    save_CT_scan_3D_image(airway_tree, origin, spacing, niigz_file_name)

    FP_voxels, FN_voxels, TP_voxels, TN_voxels = metrics

    FPR = 100 * FP_voxels / (FP_voxels + TN_voxels)
    TPR = 100 * TP_voxels / (TP_voxels + FN_voxels)     # = sensitivity
    FNR = 100 * FN_voxels / (FN_voxels + TP_voxels)
    DSC = 100 * (2 * TP_voxels) / (2 * TP_voxels + FP_voxels + FN_voxels)
    return [FPR, TPR, FNR, DSC]


def save_airway_tree_3dmodel_annotated_with_2colors(groundtruth_npy, predict_npy, origin, spacing, niigz_file_name):
    airway_tree = groundtruth_npy + predict_npy

    # Note that the colors label
    # 0              True Negative voxel                                Black
    # 1              False Positive voxel + False Negative voxel        Red
    # 2              True Positive voxel                                Green
    save_CT_scan_3D_image(airway_tree, origin, spacing, niigz_file_name)


def false_positive_rate_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    fp = np.sum(pred - pred * label) + smooth
    fpr = round(fp * 100 / (np.sum((1.0 - label)) + smooth), 3)
    return fpr

def false_negative_rate_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    fn = np.sum(label - pred * label) + smooth
    fnr = round(fn * 100 / (np.sum(label) + smooth), 3)
    return fnr

def sensitivity_calculation(pred, label):   #  identical to True-Positive-Rate
    sensitivity = round(100 - false_negative_rate_calculation(pred, label), 3)
    return sensitivity

def dice_coefficient_score_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    intersection = np.sum(pred * label)
    dice_coefficient_score = round(((2.0 * intersection + smooth) / (np.sum(pred) + np.sum(label) + smooth)) * 100, 2)
    return dice_coefficient_score

def precision_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    tp = np.sum(pred * label) + smooth
    precision = round(tp * 100 / (np.sum(pred) + smooth), 3)
    return precision

def specificity_calculation(pred, label):
    specificity = round(100 - false_positive_rate_calculation(pred, label), 3)
    return specificity

def tree_length_calculation(pred, label_skeleton, smooth=1e-5):
    pred = pred.flatten()
    label_skeleton = label_skeleton.flatten()
    tree_length = round((np.sum(pred * label_skeleton) + smooth) / (np.sum(label_skeleton) + smooth) * 100, 2)
    return tree_length

def branch_detected_calculation(pred, label_parsing, label_skeleton, thresh=0.8):
    label_branch = label_skeleton * label_parsing
    label_branch_flat = label_branch.flatten()
    label_branch_bincount = np.bincount(label_branch_flat)[1:]
    total_branch_num = label_branch_bincount.shape[0]
    pred_branch = label_branch * pred
    pred_branch_flat = pred_branch.flatten()
    pred_branch_bincount = np.bincount(pred_branch_flat)[1:]
    if total_branch_num != pred_branch_bincount.shape[0]:
        lack_num = total_branch_num - pred_branch_bincount.shape[0]
        pred_branch_bincount = np.concatenate((pred_branch_bincount, np.zeros(lack_num)))
    branch_ratio_array = pred_branch_bincount / label_branch_bincount
    branch_ratio_array = np.where(branch_ratio_array >= thresh, 1, 0)
    detected_branch_num = np.count_nonzero(branch_ratio_array)
    detected_branch_ratio = round((detected_branch_num * 100) / total_branch_num, 2)
    return total_branch_num, detected_branch_num, detected_branch_ratio