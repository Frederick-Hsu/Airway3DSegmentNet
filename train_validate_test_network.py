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


def train_network(epoch, model, data_loader, optimizer, args):
    model.train()

    start_time = time.time()

    learning_rate = get_learning_rate(epoch, args)
    assert learning_rate is not None
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    optimizer.zero_grad()

    dice_list = []  # dice coefficient values
    ppv_list = []   # positive predictive values
    accuracy_list = []
    sensitivity_list = []
    dice_hard_list = []
    loss_list = []

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

        log.warning("predicts.shape = {0}".format([predict.shape for predict in predicts]))
        log.warning("attention_mappings.shape = {0}"
                    .format([attention_mapping.shape for attention_mapping in attention_mapping_list]))

        log.warning("Under epoch #{0}/index #{1}: ".format(epoch, index))
        if args.deep_supervision:
            predict = predicts[0]
            deepsupervision6, deepsupervision7, deepsupervision8 = predicts[1], predicts[2], predicts[3]

            loss  = dice_loss(predict, label_cube)
            log.warning("dice_loss_value = {0:.5f}".format(loss.item()))
            dice_loss6 = dice_loss(deepsupervision6, label_cube)
            log.warning("dice_loss6_value = {0:.5f}".format(dice_loss6.item()))
            dice_loss7 = dice_loss(deepsupervision7, label_cube)
            log.warning("dice_loss7_value = {0:.5f}".format(dice_loss7.item()))
            dice_loss8 = dice_loss(deepsupervision8, label_cube)
            log.warning("dice_loss8_value = {0:.5f}".format(dice_loss8.item()))

            loss += dice_loss6 + dice_loss7 + dice_loss8
        else:
            predict = predicts
            loss = dice_loss(predict, label_cube)
            log.warning("dice_loss_value = {0:.5f}".format(loss.item()))

        loss += (focal_loss_value := focal_loss(predict, label_cube))
        log.warning("focal_loss_value = {0:.5f}".format(focal_loss_value.item()))
        # loss += (BCEL_value := binary_cross_entropy_loss(predict, label_cube))
        # log.warning("binary_cross_entropy_loss_value = {0:.5f}".format(BCEL_value.item()))

        if args.encoder_path_ad == True:
            # If the attention distillation was enabled in the encoder path, namely down-sampling path
            ad_gamma = [0.1, 0.1, 0.1]
            for index in range(len(ad_gamma) - 1):
                encoder_ad_loss = ad_gamma[index] * attention_distillation_loss(attention_mapping_list[index],
                                                                                attention_mapping_list[index+1],
                                                                                encoder_flag=True)
                loss += encoder_ad_loss
                log.warning("encoder_ad_loss_value = {0:.5f}".format(encoder_ad_loss.item()))
        if args.decoder_path_ad == True:
            # If the attention distillation was enabled in the decoder path, namely up-sampling path
            ad_gamma = [0.1, 0.1, 0.1]
            for index in range(3, 6):
                # Only need to consider the attention mapping6, mapping7, mapping8, mapping9
                decoder_ad_loss = ad_gamma[index-3] * attention_distillation_loss(attention_mapping_list[index],
                                                                                  attention_mapping_list[index+1],
                                                                                  encoder_flag=False)
                loss += decoder_ad_loss
                log.warning("decoder_ad_loss_value = {0:.5f}".format(decoder_ad_loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        # Calculate the segmentation metrics -------------------------------------------------------
        # Move the predict and groundtruth values from GPU to CPU, and convert to numpy type
        predict_data = predict.cpu().data.numpy()
        groundtruth_data = label_cube.cpu().data.numpy()
        groundtruth_segment_data = (groundtruth_data > binary_threshold)

        for num in range(len(batch_len)):
            dice = dice_coefficient_np(predict_data[num, 0], groundtruth_segment_data[num, 0])
            predict_segment = (predict_data > binary_threshold)
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
    mean_dice = np.mean(np.array(dice_list))
    mean_dice_hard = np.mean(np.array(dice_hard_list))
    mean_ppv = np.mean(np.array(ppv_list))
    mean_sensitivity = np.mean(np.array(sensitivity_list))
    mean_accuracy = np.mean(np.array(accuracy_list))
    mean_loss = np.mean(np.array(loss_list))

    log.warning("dice_list = {0}".format(dice_list))
    log.warning("dice_hard_list = {0}".format(dice_hard_list))
    log.warning("ppv_list = {0}".format(ppv_list))
    log.warning("sensitivity_list = {0}".format(sensitivity_list))
    log.warning("accuracy_list = {0}".format(accuracy_list))
    log.warning("loss_list = {0}".format(loss_list))

    print("Training phase, epoch = #{0}, loss = {1:.4f}, accuracy = {2:.4f}, sensitivity = {3:.4f}, "
          "dice = {4:.4f}, dice-hard = {5:.4f}, positive predictive value = {6:.4f}, "
          "elapsed-time = {7:3.2f}, learning-rate = {8:.6f}\n"
          .format(epoch, mean_loss, mean_accuracy, mean_sensitivity, mean_dice, mean_dice_hard,
                  mean_ppv, end_time - start_time, learning_rate))

    torch.cuda.empty_cache()
    return mean_loss, mean_accuracy, mean_sensitivity, mean_dice, mean_ppv


#---------------------------------------------------------------------------------------------------
def validate_network(epoch, model, data_loader, optimizer, args, save_dir):
    model.eval()
    pass
