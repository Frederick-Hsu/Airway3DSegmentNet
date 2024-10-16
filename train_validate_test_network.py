#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File  : train_validate_test_network.py
# Brief : carry out the train/validate/test actions on the network you designed.
#
#


# System's modules
import os
import time
import csv
import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage.interpolation import zoom

# User-defined modules
from log_switch import log
from various_loss_functions import dice_loss, \
                                   focal_loss, \
                                   binary_cross_entropy_loss, \
                                   attention_distillation_loss

from utils import dice_coefficient_np, positive_predictive_value_np, sensitivity_np, accuracy_np
from utils import combine_total, combine_total_avg, save_CT_scan_3D_image, normal_min_max
from Visualization.visualize import visualize_airway_tree_segment_effect, visualize_bronchus_segment_slices
from metrics_calculation import branch_detected, \
                                tree_length_detected, \
                                false_positive_rate_calculation, \
                                false_negative_rate_calculation, \
                                sensitivity_calculation, \
                                precision_calculation, \
                                dice_coefficient_score_calculation

#===================================================================================================
binary_threshold = 0.5


# Functions ========================================================================================
def get_learning_rate(epoch, args):
    if args.learning_rate is None:
        assert epoch <= args.lr_stage[-1]
        lr_stage = np.sum(epoch > args.lr_stage)
        lr = args.lr_preset[lr_stage]
    else:
        lr = args.learning_rate
    return lr


def train_network(epoch, model, data_loader, optimizer, args, tensorboard_writer):
    model.train()

    start_time = time.time()

    learning_rate = get_learning_rate(epoch, args)
    assert learning_rate is not None
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
        log.warning("In epoch #{0}, optimzer.lr = {1:.5f}".format(epoch, param_group['lr']))
    optimizer.zero_grad()

    dice_list = []  # dice coefficient values
    ppv_list = []   # positive predictive values
    accuracy_list = []
    sensitivity_list = []
    dice_hard_list = []
    loss_list = []

    pure_dice_loss_list = []
    deepsupervision6_dice_loss_list = []
    deepsupervision7_dice_loss_list = []
    deepsupervision8_dice_loss_list = []
    focal_loss_list = []
    encoder_ad_loss_list = []
    decoder_ad_loss_list = []

    log.warning("Training progress......, in epoch #{0}".format(epoch))
    # for index, (image_cube, label_cube, origin, spacing, case_name, splitID, num_DHW, shape) in enumerate(data_loader):
    for index, (image_cube, label_cube, origin, spacing, case_name, splitID, num_DHW, shape) in enumerate(tqdm(data_loader)):
        batch_len = image_cube.size(0)
        log.warning("Case names = {0}, SplitIDs = {1}, image_cube.shape = {2}, label_cube.shape = {3}, batch_size = {4}"
                    .format(case_name, splitID, image_cube.shape, label_cube.shape, batch_len))

        # Move the image_cube and label_cube tensors from CPU to GPU
        image_cube = image_cube.cuda()
        label_cube = label_cube.cuda()  # namely the ground truth in each cropped label cube
        predicts, attention_mapping_list = model(image_cube)
        # tensorboard_writer.add_graph(model, image_cube)     # Save the model's graph into the TensorBoard

        log.warning("predicts.shape = {0}".format([predict.shape for predict in predicts]))
        log.warning("attention_mappings.shape = {0}"
                    .format([attention_mapping.shape for attention_mapping in attention_mapping_list]))

        log.warning("Under epoch #{0}/index #{1}: ".format(epoch, index))
        if args.deep_supervision:
            predict = predicts[0]
            deepsupervision6, deepsupervision7, deepsupervision8 = predicts[1], predicts[2], predicts[3]

            loss  = dice_loss(predict, label_cube)
            pure_dice_loss_list.append(loss.item())

            dice_loss6 = dice_loss(deepsupervision6, label_cube)
            deepsupervision6_dice_loss_list.append(dice_loss6.item())

            dice_loss7 = dice_loss(deepsupervision7, label_cube)
            deepsupervision7_dice_loss_list.append(dice_loss7.item())

            dice_loss8 = dice_loss(deepsupervision8, label_cube)
            deepsupervision8_dice_loss_list.append(dice_loss8.item())

            loss += dice_loss6 + dice_loss7 + dice_loss8
        else:
            predict = predicts
            loss = dice_loss(predict, label_cube)
            pure_dice_loss_list.append(loss.item())

        focal_loss_value = focal_loss(predict, label_cube)
        focal_loss_list.append(focal_loss_value.item())

        loss += focal_loss_value

        if args.encoder_path_ad:
            # If the attention distillation was enabled in the encoder path, namely down-sampling path
            ad_gamma = [0.1, 0.1, 0.1]
            for index in range(len(ad_gamma) - 1):
                encoder_ad_loss = ad_gamma[index] * attention_distillation_loss(attention_mapping_list[index],
                                                                                attention_mapping_list[index+1],
                                                                                encoder_flag=True)
                encoder_ad_loss_list.append(encoder_ad_loss.item())
                loss += encoder_ad_loss
        if args.decoder_path_ad:
            # If the attention distillation was enabled in the decoder path, namely up-sampling path
            ad_gamma = [0.1, 0.1, 0.1]
            for index in range(3, 6):
                # Only need to consider the attention mapping6, mapping7, mapping8, mapping9
                decoder_ad_loss = ad_gamma[index-3] * attention_distillation_loss(attention_mapping_list[index],
                                                                                  attention_mapping_list[index+1],
                                                                                  encoder_flag=False)
                decoder_ad_loss_list.append(decoder_ad_loss.item())
                loss += decoder_ad_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        # Calculate the segmentation metrics -------------------------------------------------------
        # Move the predict and groundtruth values from GPU to CPU, and convert to numpy type
        predict_data = predict.cpu().data.numpy()
        groundtruth_data = label_cube.cpu().data.numpy()
        groundtruth_segment_data = (groundtruth_data > binary_threshold)

        for num in range(batch_len):
            dice = dice_coefficient_np(predict_data[num, 0], groundtruth_segment_data[num, 0])
            predict_segment = (predict_data[num, 0] > binary_threshold)
            dice_hard = dice_coefficient_np(predict_segment, groundtruth_segment_data[num, 0])
            ppv = positive_predictive_value_np(predict_segment, groundtruth_segment_data[num, 0])
            sensitivity = sensitivity_np(predict_segment, groundtruth_segment_data[num, 0])
            accuracy = accuracy_np(predict_segment, groundtruth_segment_data[num, 0])

            dice_list.append(dice)
            dice_hard_list.append(dice_hard)
            sensitivity_list.append(sensitivity)
            accuracy_list.append(accuracy)
            ppv_list.append(ppv)

    end_time = time.time()

    # Calculate the mean values for pure_dice_loss_list, focal_loss_list
    pure_dice_loss_mean_value = np.mean(np.array(pure_dice_loss_list))
    tensorboard_writer.add_scalar(tag="Pure dice-loss mean value in train phase",
                                  scalar_value=pure_dice_loss_mean_value,
                                  global_step=epoch)
    focal_loss_mean_value = np.mean(np.array(focal_loss_list))
    tensorboard_writer.add_scalar(tag="Focal-loss mean value in train phase",
                                  scalar_value=focal_loss_mean_value,
                                  global_step=epoch)
    # and if necessary, calculate the mean values for the belowing:
    if args.deep_supervision:
        deepsupervision6_dice_loss_mean_value = np.mean(np.array(deepsupervision6_dice_loss_list))
        tensorboard_writer.add_scalar(tag="DeepSupervision6 dice-loss mean value in train phase",
                                      scalar_value=deepsupervision6_dice_loss_mean_value,
                                      global_step=epoch)
        deepsupervision7_dice_loss_mean_value = np.mean(np.array(deepsupervision7_dice_loss_list))
        tensorboard_writer.add_scalar(tag="DeepSupervision7 dice-loss mean value in train phase",
                                      scalar_value=deepsupervision7_dice_loss_mean_value,
                                      global_step=epoch)
        deepsupervision8_dice_loss_mean_value = np.mean(np.array(deepsupervision8_dice_loss_list))
        tensorboard_writer.add_scalar(tag="DeepSupervision8 dice-loss mean value in train phase",
                                      scalar_value=deepsupervision8_dice_loss_mean_value,
                                      global_step=epoch)
    if args.encoder_path_ad:
        encoder_ad_loss_mean_value = np.mean(np.array(encoder_ad_loss_list))
        tensorboard_writer.add_scalar(tag="Encoder path AD-loss mean value in train phase",
                                      scalar_value=encoder_ad_loss_mean_value,
                                      global_step=epoch)
    if args.decoder_path_ad:
        decoder_ad_loss_mean_value = np.mean(np.array(decoder_ad_loss_list))
        tensorboard_writer.add_scalar(tag="Decoder path AD-loss mean value in train phase",
                                      scalar_value=decoder_ad_loss_mean_value,
                                      global_step=epoch)
    tensorboard_writer.flush()


    mean_dice = np.mean(np.array(dice_list))
    mean_dice_hard = np.mean(np.array(dice_hard_list))
    mean_ppv = np.mean(np.array(ppv_list))
    mean_sensitivity = np.mean(np.array(sensitivity_list))
    mean_accuracy = np.mean(np.array(accuracy_list))
    mean_loss = np.mean(np.array(loss_list))

    print("train phase, epoch = #{0}, loss = {1:.4f}, accuracy = {2:.4f}, sensitivity = {3:.4f}, "
          "dice-similarity-coefficient = {4:.4f}, dice-hard = {5:.4f}, positive-probability = {6:.4f}, "
          "elapsed-time = {7:3.2f}, learning-rate = {8:.6f}\n"
          .format(epoch, mean_loss, mean_accuracy, mean_sensitivity, mean_dice, mean_dice_hard,
                  mean_ppv, end_time - start_time, learning_rate))

    torch.cuda.empty_cache()
    return mean_loss, mean_accuracy, mean_sensitivity, mean_dice, mean_ppv


#---------------------------------------------------------------------------------------------------
def validate_test_network(epoch, phase, model, data_loader, args, save_dir, tensorboard_writer):
    model.eval()

    start_time = time.time()

    pred_total = {}
    input_total = {}
    groundtruth_total = {}

    feat6_total = {}
    feat7_total = {}
    feat8_total = {}
    feat9_total = {}

    dice_list = []
    ppv_list = []
    sensitivity_list = []
    dice_hard_list = []
    accuracy_list = []
    casename_list = []
    loss_list = []

    pure_dice_loss_list = []
    deepsupervision6_dice_loss_list = []
    deepsupervision7_dice_loss_list = []
    deepsupervision8_dice_loss_list = []
    focal_loss_list = []
    encoder_ad_loss_list = []
    decoder_ad_loss_list = []

    with torch.no_grad():
        log.warning("{1} progress......, in epoch #{0}"
                    .format(epoch, ("Validating" if phase == 'val' else 'Testing')))
        # for index, (image_cube, label_cube, origin, spacing, case_name, splitID, num_DHW, shape) in enumerate(data_loader):
        for index, (image_cube, label_cube, origin, spacing, case_name, splitID, num_DHW, shape) in enumerate(tqdm(data_loader)):
            batch_len = image_cube.size(0)
            log.warning("Case names = {0}, SplitIDs = {1}, image_cube.shape = {2}, label_cube.shape = {3}, batch_size = {4}"
                        .format(case_name, splitID, image_cube.shape, label_cube.shape, batch_len))

            case_name = case_name[0]    # Fetch out the case_name list
            splitID = splitID[0]        # Fetch out the splitID list, same as case_name

            image_cube = image_cube.cuda()
            label_cube = label_cube.cuda()
            predicts, attentions = model(image_cube)

            log.warning("predicts.shape = {0}".format([predict.shape for predict in predicts]))
            log.warning("attentions.shape = {0}".format([attention.shape for attention in attentions]))
            log.warning("Under epoch #{0}/index #{1}: ".format(epoch, index))

            if args.deep_supervision:
                predict = predicts[0]
                deepsupervision6, deepsupervision7, deepsupervision8 = predicts[1], predicts[2], predicts[3]

                loss = dice_loss(predict, label_cube)
                pure_dice_loss_list.append(loss.item())

                dice_loss6 = dice_loss(deepsupervision6, label_cube)
                deepsupervision6_dice_loss_list.append(dice_loss6.item())

                dice_loss7 = dice_loss(deepsupervision7, label_cube)
                deepsupervision7_dice_loss_list.append(dice_loss7.item())

                dice_loss8 = dice_loss(deepsupervision8, label_cube)
                deepsupervision8_dice_loss_list.append(dice_loss8.item())

                loss += dice_loss6 + dice_loss7 + dice_loss8
            else:
                predict = predicts
                loss = dice_loss(predict, label_cube)
                pure_dice_loss_list.append(loss.item())

            focal_loss_value = focal_loss(predict, label_cube)
            focal_loss_list.append(focal_loss_value.item())
            loss += focal_loss_value

            if args.encoder_path_ad:
                ad_gamma = [0.1, 0.1, 0.1]
                for index in range(len(ad_gamma) - 1):  # attentions 0, 1, 2
                    encoder_ad_loss = ad_gamma[index] * attention_distillation_loss(attentions[index],
                                                                                    attentions[index + 1],
                                                                                    encoder_flag=True)
                    encoder_ad_loss_list.append(encoder_ad_loss.item())
                    loss += encoder_ad_loss
            if args.decoder_path_ad:
                ad_gamma = [0.1, 0.1, 0.1]
                for index in range(3, 6):   # attentions 3, 4, 5, 6
                    decoder_ad_loss = ad_gamma[index - 3] * attention_distillation_loss(attentions[index],
                                                                                        attentions[index + 1],
                                                                                        encoder_flag=False)
                    decoder_ad_loss_list.append(decoder_ad_loss.item())
                    loss += decoder_ad_loss

            loss_list.append(loss.item())
            #---------------------------------------------------------------------------------------
            predict_data = predict.cpu().data.numpy()
            gt_data = label_cube.cpu().data.numpy()     # the ground-truth data
            gt_seg_data = (gt_data > binary_threshold)  # the ground-truth segmentation data
            image_data = image_cube.cpu().data.numpy()  # Move the image_cube from GPU to CPU, then convert to numpy
            origin_data = origin.numpy()
            spacing_data = spacing.numpy()

            feat6 = attentions[3].cpu().data.numpy()    # feat6 <---> attention6
            feat7 = attentions[4].cpu().data.numpy()    # feat7 <---> attention7
            feat8 = attentions[5].cpu().data.numpy()    # feat8 <---> attention8
            feat9 = attentions[6].cpu().data.numpy()    # feat9 <---> attention9

            for num in range(batch_len):
                curr_img_data = (image_data[num, 0] * 255)
                curr_gt_data = gt_seg_data[num, 0]
                curr_predict_data = predict_data[num, 0]
                curr_origin = origin_data[num].tolist()
                curr_spacing = spacing_data[num].tolist()
                curr_splitID = int(splitID[num])
                assert (curr_splitID >= 0)
                curr_name = case_name[num]
                curr_num_DHW = num_DHW[num]
                curr_shape = shape[num]

                if curr_name not in input_total.keys():
                    input_total[curr_name] = []
                if curr_name not in groundtruth_total.keys():
                    groundtruth_total[curr_name] = []
                if curr_name not in pred_total.keys():
                    pred_total[curr_name] = []
                if curr_name not in feat6_total.keys():
                    feat6_total[curr_name] = []
                if curr_name not in feat7_total.keys():
                    feat7_total[curr_name] = []
                if curr_name not in feat8_total.keys():
                    feat8_total[curr_name] = []
                if curr_name not in feat9_total.keys():
                    feat9_total[curr_name] = []

                curr_input_info = [curr_img_data,     curr_splitID, curr_num_DHW, curr_shape, curr_origin, curr_spacing]
                curr_gt_info    = [curr_gt_data,      curr_splitID, curr_num_DHW, curr_shape, curr_origin, curr_spacing]
                curr_pred_info  = [curr_predict_data, curr_splitID, curr_num_DHW, curr_shape, curr_origin, curr_spacing]

                input_total[curr_name].append(curr_input_info)
                groundtruth_total[curr_name].append(curr_gt_info)
                pred_total[curr_name].append(curr_pred_info)

                if args.save_feature:
                    curr_feat6 = zoom(feat6[num, 0], 8, order=0, mode='nearest')
                    curr_feat7 = zoom(feat7[num, 0], 4, order=0, mode='nearest')
                    curr_feat8 = zoom(feat8[num, 0], 2, order=0, mode='nearest')
                    curr_feat9 = feat9[num, 0]

                    curr_feat6_info = [curr_feat6, curr_splitID, curr_num_DHW, curr_shape, curr_origin, curr_spacing]
                    curr_feat7_info = [curr_feat7, curr_splitID, curr_num_DHW, curr_shape, curr_origin, curr_spacing]
                    curr_feat8_info = [curr_feat8, curr_splitID, curr_num_DHW, curr_shape, curr_origin, curr_spacing]
                    curr_feat9_info = [curr_feat9, curr_splitID, curr_num_DHW, curr_shape, curr_origin, curr_spacing]

                    feat6_total[curr_name].append(curr_feat6_info)
                    feat7_total[curr_name].append(curr_feat7_info)
                    feat8_total[curr_name].append(curr_feat8_info)
                    feat9_total[curr_name].append(curr_feat9_info)

    # Combine all these cases together
    stride = args.val_stride
    cube_size = args.val_cube_size

    pure_dice_loss_mean_value = np.mean(np.array(pure_dice_loss_list))
    tensorboard_writer.add_scalar(tag="Pure dice-loss mean value in {0} phase".format(phase),
                                  scalar_value=pure_dice_loss_mean_value,
                                  global_step=epoch)
    focal_loss_mean_value = np.mean(np.array(focal_loss_list))
    tensorboard_writer.add_scalar(tag="Focal-loss mean value in {0} phase".format(phase),
                                  scalar_value=focal_loss_mean_value,
                                  global_step=epoch)
    if args.deep_supervision:
        deepsupervision6_dice_loss_mean_value = np.mean(np.array(deepsupervision6_dice_loss_list))
        tensorboard_writer.add_scalar(tag="DeepSupervision6 dice-loss mean value in {0} phase".format(phase),
                                      scalar_value=deepsupervision6_dice_loss_mean_value,
                                      global_step=epoch)
        deepsupervision7_dice_loss_mean_value = np.mean(np.array(deepsupervision7_dice_loss_list))
        tensorboard_writer.add_scalar(tag="DeepSupervision7 dice-loss mean value in {0} phase".format(phase),
                                      scalar_value=deepsupervision7_dice_loss_mean_value,
                                      global_step=epoch)
        deepsupervision8_dice_loss_mean_value = np.mean(np.array(deepsupervision8_dice_loss_list))
        tensorboard_writer.add_scalar(tag="DeepSupervision8 dice-loss mean value in {0} phase".format(phase),
                                      scalar_value=deepsupervision8_dice_loss_mean_value,
                                      global_step=epoch)
    if args.encoder_path_ad:
        encoder_ad_loss_mean_value = np.mean(np.array(encoder_ad_loss_list))
        tensorboard_writer.add_scalar(tag="Encoder path AD-loss mean value in {0} phase".format(phase),
                                      scalar_value=encoder_ad_loss_mean_value,
                                      global_step=epoch)
    if args.decoder_path_ad:
        decoder_ad_loss_mean_value = np.mean(np.array(decoder_ad_loss_list))
        tensorboard_writer.add_scalar(tag="Decoder path AD-loss mean value in {0} phase".format(phase),
                                      scalar_value=decoder_ad_loss_mean_value,
                                      global_step=epoch)
    tensorboard_writer.flush()

    for curr_name in input_total.keys():
        curr_input = input_total[curr_name]
        curr_groundtruth = groundtruth_total[curr_name]
        curr_pred = pred_total[curr_name]

        # The raw CT 3D image does not need to combine
        input_combine, input_origin, input_spacing = combine_total(curr_input, stride, cube_size)

        # Combine the label cubes
        label_combine, curr_origin, curr_spacing = combine_total(curr_groundtruth, stride, cube_size)
        # Combine the predict cubes
        pred_combine, pred_origin, pred_spacing = combine_total_avg(curr_pred, stride, cube_size)
        pred_combine_binarythreshold = (pred_combine > binary_threshold)

        curr_label_path = os.path.join(save_dir, "{0}-groundtruth.nii.gz".format(curr_name))
        curr_pred_path  = os.path.join(save_dir, "{0}-predict.nii.gz".format(curr_name))
        save_CT_scan_3D_image(label_combine.astype(dtype='uint8'), curr_origin, curr_spacing, curr_label_path)
        save_CT_scan_3D_image(pred_combine_binarythreshold.astype(dtype='uint8'), curr_origin, curr_spacing, curr_pred_path)

        visualize_airway_tree_segment_effect(epoch_num=epoch,
                                             phase=phase,
                                             case_name=curr_name,
                                             label_cuboid=label_combine,
                                             segment_cuboid=pred_combine_binarythreshold,
                                             origin=curr_origin,
                                             spacing=curr_spacing,
                                             save_dir=save_dir,
                                             tensorboard_writer=tensorboard_writer)

        visualize_bronchus_segment_slices(epoch_num=epoch,
                                          phase=phase,
                                          case_name=curr_name,
                                          raw_image_cuboid=input_combine,
                                          label_cuboid=label_combine,
                                          segment_cuboid=pred_combine_binarythreshold,
                                          save_dir=save_dir,
                                          tensorboard_writer=tensorboard_writer)

        # Calculate the metrics and save them to tensor board
        FPR = false_positive_rate_calculation(pred=pred_combine_binarythreshold, label=label_combine)
        FNR = false_negative_rate_calculation(pred=pred_combine_binarythreshold, label=label_combine)
        Sensitivity = sensitivity_calculation(pred=pred_combine_binarythreshold, label=label_combine)
        Precision = precision_calculation(pred=pred_combine_binarythreshold, label=label_combine)
        DSC = dice_coefficient_score_calculation(pred=pred_combine_binarythreshold, label=label_combine)
        BD = branch_detected(predict=pred_combine_binarythreshold, groundtruth=label_combine)
        TD = tree_length_detected(predict=pred_combine_binarythreshold, groundtruth=label_combine)

        tensorboard_writer.add_scalar(tag="{0}: False Positive Rate at {1} phase".format(curr_name, phase),
                                      scalar_value=FPR,
                                      global_step=epoch)
        tensorboard_writer.add_scalar(tag="{0}: False Negative Rate at {1} phase".format(curr_name, phase),
                                      scalar_value=FNR,
                                      global_step=epoch)
        tensorboard_writer.add_scalar(tag="{0}: Sensitivity = True Positive Rate at {1} phase".format(curr_name, phase),
                                      scalar_value=Sensitivity,
                                      global_step=epoch)
        tensorboard_writer.add_scalar(tag="{0}: Precision at {1} phase".format(curr_name, phase),
                                      scalar_value=Precision,
                                      global_step=epoch)
        tensorboard_writer.add_scalar(tag="{0}: DSC at {1} phase".format(curr_name, phase),
                                      scalar_value=DSC,
                                      global_step=epoch)
        tensorboard_writer.add_scalar(tag="{0}: Branch Detected Rate at {1} phase".format(curr_name, phase),
                                      scalar_value=BD,
                                      global_step=epoch)
        tensorboard_writer.add_scalar(tag="{0}: Tree Length Detected at {1} phase".format(curr_name, phase),
                                      scalar_value=TD,
                                      global_step=epoch)
        tensorboard_writer.flush()

        #-------------------------------------------------------------------------------------------
        if args.save_feature:
            curr_feat6 = feat6_total[curr_name]
            curr_feat7 = feat7_total[curr_name]
            curr_feat8 = feat8_total[curr_name]
            curr_feat9 = feat9_total[curr_name]

            feat6, feat_origin, feat_spacing = combine_total_avg(curr_feat6, stride, cube_size)
            feat7, _, _ = combine_total_avg(curr_feat7, stride, cube_size)
            feat8, _, _ = combine_total_avg(curr_feat8, stride, cube_size)
            feat9, _, _ = combine_total_avg(curr_feat9, stride, cube_size)
            feat6 = normal_min_max(feat6) * 255
            feat7 = normal_min_max(feat7) * 255
            feat8 = normal_min_max(feat8) * 255
            feat9 = normal_min_max(feat9) * 255

            curr_feat6_path = os.path.join(save_dir, "{0}-feat6.nii.gz".format(curr_name))
            curr_feat7_path = os.path.join(save_dir, "{0}-feat7.nii.gz".format(curr_name))
            curr_feat8_path = os.path.join(save_dir, "{0}-feat8.nii.gz".format(curr_name))
            curr_feat9_path = os.path.join(save_dir, "{0}-feat9.nii.gz".format(curr_name))
            save_CT_scan_3D_image(feat6.astype(dtype='uint8'), curr_origin, curr_spacing, curr_feat6_path)
            save_CT_scan_3D_image(feat7.astype(dtype='uint8'), curr_origin, curr_spacing, curr_feat7_path)
            save_CT_scan_3D_image(feat8.astype(dtype='uint8'), curr_origin, curr_spacing, curr_feat8_path)
            save_CT_scan_3D_image(feat9.astype(dtype='uint8'), curr_origin, curr_spacing, curr_feat9_path)

            feat6_cuboid_np = feat6.astype(dtype='uint8')
            feat7_cuboid_np = feat7.astype(dtype='uint8')
            feat8_cuboid_np = feat8.astype(dtype='uint8')
            feat9_cuboid_np = feat9.astype(dtype='uint8')
            feat6_airway_top_view = np.sum(feat6_cuboid_np, axis=1)
            feat7_airway_top_view = np.sum(feat7_cuboid_np, axis=1)
            feat8_airway_top_view = np.sum(feat8_cuboid_np, axis=1)
            feat9_airway_top_view = np.sum(feat9_cuboid_np, axis=1)

            tensorboard_writer.add_image(tag="{0} epoch #{1}, {2}: attention distillation mapping6".format(phase, epoch, curr_name),
                                         img_tensor=np.flipud(feat6_airway_top_view),
                                         global_step=epoch,
                                         dataformats='HW')
            tensorboard_writer.add_image(tag="{0} epoch #{1}, {2}: attention distillation mapping7".format(phase, epoch, curr_name),
                                         img_tensor=np.flipud(feat7_airway_top_view),
                                         global_step=epoch,
                                         dataformats='HW')
            tensorboard_writer.add_image(tag="{0} epoch #{1}, {2}: attention distillation mapping8".format(phase, epoch, curr_name),
                                         img_tensor=np.flipud(feat8_airway_top_view),
                                         global_step=epoch,
                                         dataformats='HW')
            tensorboard_writer.add_image(tag="{0} epoch #{1}, {2}: attention distillation mapping9".format(phase, epoch, curr_name),
                                         img_tensor=np.flipud(feat9_airway_top_view),
                                         global_step=epoch,
                                         dataformats='HW')
            tensorboard_writer.flush()

        # -------------------------------------------------------------------------------------------
        log.warning("{0} case:".format(curr_name))
        curr_dice_hard = dice_coefficient_np(pred_combine_binarythreshold, label_combine)
        log.warning("\tdice_hard = {0:.5f}".format(curr_dice_hard))
        curr_dice = dice_coefficient_np(pred_combine, label_combine)
        log.warning("\tdice = {0:.5f}".format(curr_dice))
        curr_ppv = positive_predictive_value_np(pred_combine_binarythreshold, label_combine)
        log.warning("\tpositive_probability = {0:.5f}".format(curr_ppv))
        curr_sensitivity = sensitivity_np(pred_combine_binarythreshold, label_combine)
        log.warning("\tsensitivity = {0:.5f}".format(curr_sensitivity))
        curr_accuracy = accuracy_np(pred_combine_binarythreshold, label_combine)
        log.warning("\taccuracy = {0:.5f}".format(curr_accuracy))

        dice_list.append(curr_dice)
        dice_hard_list.append(curr_dice_hard)
        ppv_list.append(curr_ppv)
        accuracy_list.append(curr_accuracy)
        sensitivity_list.append(curr_sensitivity)
        casename_list.append(curr_name)

        del curr_groundtruth, curr_pred, label_combine, pred_combine_binarythreshold, pred_combine

    end_time = time.time()

    # Save [accuracy, sensitivity, dice, ppv] into a specified csv file.
    save_metrics_into_csvfile(save_dir,
                              "{0}_results_epoch{1}.csv".format(phase, epoch),
                              *[casename_list, accuracy_list, sensitivity_list, dice_list, ppv_list])

    mean_dice = np.mean(np.array(dice_list))
    mean_dice_hard = np.mean(np.array(dice_hard_list))
    mean_ppv = np.mean(np.array(ppv_list))
    mean_sensitivity = np.mean(np.array(sensitivity_list))
    mean_accuracy = np.mean(np.array(accuracy_list))
    mean_loss = np.mean(np.array(loss_list))

    print("{0} phase, epoch = #{1}, loss = {2:.4f}, accuracy = {3:.4f}, sensitivity = {4:.4f}, "
          "dice-similarity-coefficient = {5:.4f}, dice-hard = {6:.4f}, positive-probability = {7:.4f}, "
          "elapsed-time = {8:3.2f}\n"
          .format(phase, epoch, mean_loss, mean_accuracy, mean_sensitivity, mean_dice, mean_dice_hard,
                  mean_ppv, end_time - start_time))

    torch.cuda.empty_cache()
    return mean_loss, mean_accuracy, mean_sensitivity, mean_dice, mean_ppv


#---------------------------------------------------------------------------------------------------
def save_metrics_into_csvfile(save_dir, csv_file_name, *metrics):
    all_results = {'ATM22': []}
    casename_list = metrics[0]
    accuracy_list = metrics[1]
    sensitivity_list = metrics[2]
    dice_list = metrics[3]
    ppv_list = metrics[4]

    with open(os.path.join(save_dir, csv_file_name), 'w') as fh:
        writer = csv.writer(fh)

        title_row = ['name', 'accuracy', 'sensitivity', 'dice', 'positive probability value']
        writer.writerow(title_row)

        for index in range(len(casename_list)):
            casename = casename_list[index]

            row = [casename,
                   float(round(accuracy_list[index] * 100, 3)),
                   float(round(sensitivity_list[index] * 100, 3)),
                   float(round(dice_list[index] * 100, 3)),
                   float(round(ppv_list[index] * 100, 3))]
            all_results['ATM22'].append([row[1], row[2], row[3], row[4]])
            writer.writerow(row)

        atm22_results_mean_value = np.mean(np.array(all_results['ATM22']), axis=0)
        atm22_results_std_deviation = np.std(np.array(all_results['ATM22']), axis=0)

        atm22_mean = ['ATM22 mean value',
                      atm22_results_mean_value[0],      # accuracy mean value
                      atm22_results_mean_value[1],      # sensitivity mean value
                      atm22_results_mean_value[2],      # dice mean value
                      atm22_results_mean_value[3]]      # ppv mean value
        atm22_std_deviation = ['ATM22 std deviation',
                               atm22_results_std_deviation[0],  # accuracy std deviation
                               atm22_results_std_deviation[1],  # sensitivity std deviation
                               atm22_results_std_deviation[2],  # dice std deviation
                               atm22_results_std_deviation[3]]  # ppv std deviation
        writer.writerow(atm22_mean)
        writer.writerow(atm22_std_deviation)
