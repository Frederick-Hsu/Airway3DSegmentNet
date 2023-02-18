#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File  : metrics_calculation.py
#
#


import os
import numpy as np
from skimage.morphology import skeletonize_3d
from scipy import ndimage
import skimage.measure as measure
import SimpleITK as sitk

from utils import save_CT_scan_3D_image, load_CT_scan_3D_image
from log_switch import log

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

def large_connected_domain(label):
    cd, num = measure.label(label, return_num = True, connectivity=1)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd==(k+1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    #print(volume_sort)
    label = (cd==(volume_sort[-1]+1)).astype(np.uint8)
    label = ndimage.binary_fill_holes(label)
    label = label.astype(np.uint8)
    return label

def skeleton_parsing(skeleton):
    #separate the skeleton
    neighbor_filter = ndimage.generate_binary_structure(3, 3)
    skeleton_filtered = ndimage.convolve(skeleton, neighbor_filter) * skeleton
    #distribution = skeleton_filtered[skeleton_filtered>0]
    #plt.hist(distribution)
    skeleton_parse = skeleton.copy()
    skeleton_parse[skeleton_filtered>3] = 0
    con_filter = ndimage.generate_binary_structure(3, 3)
    cd, num = ndimage.label(skeleton_parse, structure = con_filter)
    #remove small branches
    for i in range(num):
        a = cd[cd==(i+1)]
        if a.shape[0]<5:
            skeleton_parse[cd==(i+1)] = 0
    cd, num = ndimage.label(skeleton_parse, structure = con_filter)
    return skeleton_parse, cd, num

def tree_parsing_func(skeleton_parse, label, cd):
    #parse the airway tree
    edt, inds = ndimage.distance_transform_edt(1-skeleton_parse, return_indices=True)
    tree_parsing = np.zeros(label.shape, dtype = np.uint16)
    tree_parsing = cd[inds[0,...], inds[1,...], inds[2,...]] * label
    return tree_parsing

def loc_trachea(tree_parsing, num):
    #find the trachea
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((tree_parsing==(k+1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    #print(volume_sort)
    trachea = (volume_sort[-1]+1)
    return trachea

def find_bb_3D(label):
    if len(label.shape) != 3:
        log.warning("The dimension of input is not 3!")
        os._exit()
    sum_x = np.sum(label, axis = (1,2))
    sum_y = np.sum(label, axis = (0,2))
    sum_z = np.sum(label, axis = (0,1))
    xf = np.where(sum_x)
    xf = xf[0]
    yf = np.where(sum_y)
    yf = yf[0]
    zf = np.where(sum_z)
    zf = zf[0]
    x_length = xf.max() - xf.min() + 1
    y_length = yf.max() - yf.min() + 1
    z_length = zf.max() - zf.min() + 1
    x1 = xf.min()
    y1 = yf.min()
    z1 = zf.min()
    
    cs = [x_length+8, y_length+8, z_length+8]
    for j in range(3):
        if cs[j]>label.shape[j]:
            cs[j] = label.shape[j]
    #print(cs[0], x_length)
    #x_length, y_length, z_length, x1, y1, z1 = find_bb_3D(label2)
    cs = np.array(cs, dtype=np.uint16)
    size = label.shape
    xl = x1 - (cs[0]-x_length)//2
    yl = y1 - (cs[1]-y_length)//2
    zl = z1 - (cs[2] - z_length)//2
    xr = xl + cs[0]
    yr = yl + cs[1]
    zr = zl + cs[2]
    if xl < 0:
        xl = 0
        xr = cs[0]
    if xr > size[0]:
        xr = size[0]
        xl = xr - cs[0]
    if yl < 0:
        yl = 0
        yr = cs[1]
    if yr > size[1]:
        yr = size[1]
        yl = yr - cs[1]
    if zl < 0:
        zl = 0
        zr = cs[2]
    if zr > size[2]:
        zr = size[2]
        zl = zr - cs[2]
    return xl, xr, yl, yr, zl, zr

def adjacent_map(tree_parsing, num):
    #build the adjacency matrix
    ad_matric = np.zeros((num, num), dtype=np.uint8)
    #i = 1
    for i in range(num):
        cd_cur = (tree_parsing==(i+1)).astype(np.uint8)
        xl, xr, yl, yr, zl, zr = find_bb_3D(cd_cur)
        cd_cur = cd_cur[xl:xr, yl:yr, zl:zr]
        #edt = ndimage.distance_transform_edt(1-cd_cur, return_indices=False)
        dilation_filter = ndimage.generate_binary_structure(3, 1)
        boundary = ndimage.binary_dilation(cd_cur, structure=dilation_filter).astype(cd_cur.dtype) - cd_cur
        adjacency = boundary*tree_parsing[xl:xr, yl:yr, zl:zr]
        adjacency_elements = np.unique(adjacency[adjacency>0])
        for j in range(len(adjacency_elements)):
            ad_matric[i,adjacency_elements[j]-1] = 1
    return ad_matric

def parent_children_map(ad_matric, trachea, num):
    #build the parent map and children map
    parent_map = np.zeros((num, num), dtype=np.uint8)
    children_map = np.zeros((num, num), dtype=np.uint8)
    generation = np.zeros((num), dtype=np.uint8)
    processed = np.zeros((num), dtype=np.uint8)
    
    processing = [trachea-1]
    parent_map[trachea-1, trachea-1] = 1
    while len(processing)>0:
        iteration = processing
        log.warning("items in this iteration: ", iteration)
        processed[processing] = 1
        processing = []
        while len(iteration)>0:
            cur = iteration.pop()
            children = np.where(ad_matric[cur,:]>0)[0]
            for i in range(len(children)):
                cur_child = children[i]
                if parent_map[cur_child,:].sum()==0:
                    parent_map[cur_child, cur] = 1
                    children_map[cur, cur_child] = 1
                    generation[cur_child] = generation[cur] + 1
                    processing.append(cur_child)
                else:
                    if generation[cur]+1 == generation[cur_child]:
                        parent_map[cur_child, cur] = 1
                        children_map[cur, cur_child] = 1
    return parent_map, children_map, generation

def whether_refinement(parent_map, children_map, tree_parsing, num, trachea):
    witem = np.sum(parent_map, axis=1)
    witems = np.where(witem>1)[0]
    child_num = np.sum(children_map, axis=1)
    problem1_loc = np.where(child_num==1)[0]
    
    #First, fuse the parents of one child
    delete_ids = []
    if len(witems)>0:
        for i in range(len(witems)):
            log.warning("item: ", witems[i], "parents: ", np.where(parent_map[witems[i],:]>0)[0])
            cur_witem = np.where(parent_map[witems[i],:]>0)[0]
            for j in range(1, len(cur_witem)):
                tree_parsing[tree_parsing==(cur_witem[j]+1)] = cur_witem[0]+1
                if cur_witem[j] not in delete_ids:
                    delete_ids.append(cur_witem[j])
    
    #second, delete the alone child
    for i in range(len(problem1_loc)):
        cur_loc = problem1_loc[i]
        if cur_loc not in delete_ids:
            cur_child = np.where(children_map[cur_loc,:]==1)[0][0]
            if cur_child not in delete_ids:
                tree_parsing[tree_parsing==(cur_child+1)] = cur_loc+1
                delete_ids.append(cur_child)
                
    # =============================================================================
    # Third, delete the wrong trachea blocks
    # Tchildren = np.where(children_map[trachea-1,:]>0)[0]
    # z_trachea = np.mean(np.where(cd==(trachea))[0])
    # for i in range(len(Tchildren)):
    #     z_child = np.mean(np.where(cd==(Tchildren[i]+1))[0])
    #     if z_child > z_trachea:
    #         if Tchildren[i] not in delete_ids:
    #             tree_parsing[tree_parsing==(Tchildren[i]+1)] = trachea
    #             delete_ids.append(Tchildren[i])
    # =============================================================================
                
    if len(delete_ids) == 0:
        return False
    else:
        return True

def tree_refinement(parent_map, children_map, tree_parsing, num, trachea):
    witem = np.sum(parent_map, axis=1)
    witems = np.where(witem>1)[0]
    if len(witems)>0:
        for i in range(len(witems)):
            log.warning("item: ", witems[i], "parents: ", np.where(parent_map[witems[i],:]>0)[0])
       
    #print(np.where(children_map[160,:]>0)[0])
    child_num = np.sum(children_map, axis=1)
    problem1_loc = np.where(child_num==1)[0]
    
    #First, fuse the parents of one child
    delete_ids = []
    if len(witems)>0:
        for i in range(len(witems)):
            log.warning("item: ", witems[i], "parents: ", np.where(parent_map[witems[i],:]>0)[0])
            cur_witem = np.where(parent_map[witems[i],:]>0)[0]
            for j in range(1, len(cur_witem)):
                tree_parsing[tree_parsing==(cur_witem[j]+1)] = cur_witem[0]+1
                if cur_witem[j] not in delete_ids:
                    delete_ids.append(cur_witem[j])
    
    #second, delete the only child
    for i in range(len(problem1_loc)):
        cur_loc = problem1_loc[i]
        if cur_loc not in delete_ids:
            cur_child = np.where(children_map[cur_loc,:]==1)[0][0]
            if cur_child not in delete_ids:
                tree_parsing[tree_parsing==(cur_child+1)] = cur_loc+1
                delete_ids.append(cur_child)
                
    # =============================================================================
    # Third, delete the wrong trachea blocks
    # Tchildren = np.where(children_map[trachea-1,:]>0)[0]
    # z_trachea = np.mean(np.where(cd==(trachea))[0])
    # for i in range(len(Tchildren)):
    #     z_child = np.mean(np.where(cd==(Tchildren[i]+1))[0])
    #     if z_child > z_trachea:
    #         if Tchildren[i] not in delete_ids:
    #             tree_parsing[tree_parsing==(Tchildren[i]+1)] = trachea
    #             delete_ids.append(Tchildren[i])
    # =============================================================================
                
    #delete the problematic blocks from the tree
    for i in range(num):
        if i not in delete_ids:
            move = len(np.where(np.array(delete_ids)<i)[0])
            tree_parsing[tree_parsing==(i+1)] = i+1-move
    num = num - len(delete_ids) 
    
    return tree_parsing, num

def tree_length_detected(predict, groundtruth):
    skeleton = skeletonize_3d(groundtruth)
    skeleton = (skeleton > 0).astype(np.uint8)
    return tree_length_calculation(predict, skeleton)

def branch_detected(predict, groundtruth):
    skeleton = skeletonize_3d(groundtruth)
    skeleton = (skeleton > 0).astype(np.uint8)
    
    lcd = large_connected_domain(groundtruth)
    lcd_3d_skeleton = skeletonize_3d(lcd)

    parsed_skeleton, skeleton_connected_region, num = skeleton_parsing(lcd_3d_skeleton)
    airway_tree = tree_parsing_func(parsed_skeleton, lcd, skeleton_connected_region)

    # Extract the trachea from entire airway tree
    trachea = loc_trachea(airway_tree, num)

    # Build the adjacent matrix from airway tree
    adjacent_matrix = adjacent_map(airway_tree, num)

    # Find out the parent-tree and child-tree, how many generate from parent to children
    parent_tree, child_tree, generation = parent_children_map(adjacent_matrix, trachea, num)

    # Continous refine the airway tree, until no new branch tracked
    while whether_refinement(parent_tree, child_tree, airway_tree, num, trachea) is True:
        airway_tree, num = tree_refinement(parent_tree, child_tree, airway_tree, num, trachea)
        trachea = loc_trachea(airway_tree, num)
        adjacent_matrix = adjacent_map(airway_tree, num)
        parent_tree, child_tree, generation = parent_children_map(adjacent_matrix, trachea, num)
    
    branch_list = branch_detected_calculation(predict,
                                              label_parsing=airway_tree,
                                              label_skeleton=skeleton)
    
    total_branches, detected_branches, detected_branches_ratio = branch_list
    return detected_branches_ratio

#===================================================================================================
if __name__ == "__main__":
    # ct_image_file = "metrics_calculation/ATM_054_0000_clean_hu.nii.gz"
    groundtruth_file = "metrics_calculation/ATM_054_0000-groundtruth.nii.gz"
    segment_predict_file = "metrics_calculation/ATM_054_0000-predict.nii.gz"
    
    # image_npy, origin, spacing = load_CT_scan_3D_image(ct_image_file)
    groundtruth_npy, origin, spacing = load_CT_scan_3D_image(groundtruth_file)
    segment_pred_npy, _, _ = load_CT_scan_3D_image(segment_predict_file)
    
    save_airway_tree_3dmodel_annotated_with_3colors(groundtruth_npy, segment_pred_npy, origin, spacing,
                                                    "ATM_054_0000_airway_tree_segmentation_with_3colors.nii.gz")
    save_airway_tree_3dmodel_annotated_with_2colors(groundtruth_npy, segment_pred_npy, origin, spacing,
                                                    "ATM_054_0000_airway_tree_segmentation_with_2colors.nii.gz")
    
    FalsePositiveRate = false_positive_rate_calculation(segment_pred_npy, groundtruth_npy)
    FalseNegativeRate = false_negative_rate_calculation(segment_pred_npy, groundtruth_npy)
    Sensitivity = sensitivity_calculation(segment_pred_npy, groundtruth_npy)
    Precision = precision_calculation(segment_pred_npy, groundtruth_npy)
    DiceSimilarityCoefficient = dice_coefficient_score_calculation(segment_pred_npy, groundtruth_npy)
    TreeLengthDetected = tree_length_detected(segment_pred_npy, groundtruth_npy)
    BranchDetected = branch_detected(segment_pred_npy, groundtruth_npy)
    
    print("Metrics: FPR = {0}%, FNR = {1}%, Sensitivity = {2}%, DSC = {3}, Precision = {4}%, TLD = {5}%, BD = {6}%"
          .format(FalsePositiveRate, FalseNegativeRate, Sensitivity, DiceSimilarityCoefficient, 
                  Precision, TreeLengthDetected, BranchDetected))
    
    