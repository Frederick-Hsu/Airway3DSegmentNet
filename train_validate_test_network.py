#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File  : train_validate_test_network.py
# Brief : carry out the train/validate/test actions on the network you designed.
#
#


# System's modules
import time
import torch
from tqdm import tqdm

# User-defined modules
from log_switch import log
from various_loss_functions import dice_loss, \
                                   focal_loss, \
                                   binary_cross_entropy_loss, \
                                   attention_distillation_loss


# Functions ========================================================================================
def train_network(epoch, model, data_loader, optimizer, args, save_dir):
    model.train()

    start_time = time.time()
    cube_size = args.train_cube_size
    stride = args.train_stride

    optimizer.zero_grad()
    log.warning("Training progress......, in epoch #{0}".format(epoch))
    # for index, (image_cube, label_cube, origin, spacing, case_name, splitID, num_DHW, shape) in enumerate(data_loader):
    for index, (image_cube, label_cube, origin, spacing, case_name, splitID, num_DHW, shape) in enumerate(tqdm(data_loader)):
        log.warning("Case names = {0}, SplitIDs = {1}, image_cube.shape = {2}, label_cube.shape = {3}, batch_size = {4}"
                    .format(case_name, splitID, image_cube.shape, label_cube.shape, image_cube.size(0)))

        # Move the image_cube and label_cube tensors from CPU to GPU
        image_cube = image_cube.cuda()
        label_cube = label_cube.cuda()  # namely the ground truth in each cropped label cube
        predicts, attention_mapping_list = model(image_cube)

        if args.deep_supervision:
            predict = predicts[0]
            deepsupervision6, deepsupervision7, deepsupervision8 = predicts[1], predicts[2], predicts[3]
            loss = dice_loss(predict, label_cube) + \
                   dice_loss(deepsupervision6, label_cube) + \
                   dice_loss(deepsupervision7, label_cube) + \
                   dice_loss(deepsupervision8, label_cube)
        else:
            predict = predicts
            loss = dice_loss(predict, label_cube)

        loss += focal_loss(predict, label_cube)
        # loss += binary_cross_entropy_loss(predict, label_cube)

        if args.encoder_path_ad == True:
            # If the attention distillation was enabled in the encoder path, namely down-sampling path
            ad_gamma = [0.1, 0.1, 0.1]
            for index in range(len(ad_gamma) - 1):
                loss += ad_gamma[index] * attention_distillation_loss(attention_mapping_list[index],
                                                                      attention_mapping_list[index+1],
                                                                      encoder_flag=True)
        if args.decoder_path_ad == True:
            # If the attention distillation was enabled in the decoder path, namely up-sampling path
            ad_gamma = [0.1, 0.1, 0.1]
            for index in range(3, 6):
                # Only need to consider the attention mapping6, mapping7, mapping8, mapping9
                loss += ad_gamma[index-3] * attention_distillation_loss(attention_mapping_list[index],
                                                                        attention_mapping_list[index+1],
                                                                        encoder_flag=False)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #-------------------------------------------------------------------------------------------


    end_time = time.time()
    pass


#---------------------------------------------------------------------------------------------------
def validate_network(epoch, model, data_loader, optimizer, args, save_dir):
    model.eval()
    pass
