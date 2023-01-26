#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : various_loss_functions.py
# Brief     : Define some different loss functions in this file
#
#


import torch
import torch.nn.functional as F

# hyper-parameters:
smooth = 1.0
epsilon = 1e-06

# Functions ========================================================================================
def dice_loss(predict, groundtruth):
    r'''
    Dice Similarity Coefficient loss

    Parameters
    ----------
    predict     : the neural-network-model computes out the predictive values
    groundtruth : the ground-truth values

    Returns
    -------
    the DSC value
    '''
    iflatten = predict.view(-1)
    tflatten = groundtruth.view(-1)
    intersection = torch.sum(iflatten * tflatten)
    return 1.0 - (2.0 * intersection + smooth) / (torch.sum(iflatten) + torch.sum(tflatten) + smooth)


#---------------------------------------------------------------------------------------------------
def attention_distillation_loss(predict, groundtruth, encoder_flag=True):
    r'''
    Attention distillation loss

    :param predict: input prediction
    :param groundtruth: input ground-truth
    :param encoder_flag: True to enable the encoder-path AD,
                         False to enable the decoder-path AD
    '''
    groundtruth = groundtruth.detach()
    if  (groundtruth.size(-1) == predict.size(-1)) and \
        (groundtruth.size(-2) == predict.size(-2)):
        # ground-truth and prediction have the same spatial resolution
        pass
    else:
        if encoder_flag == True:
            # groundtruth is smaller than the predict, use consecutive layers with scale factor 2
            groundtruth = F.interpolate(groundtruth, scale_factor=2, mode='trilinear')
        else:
            # groundtruth is bigger than the predict, use consecutive layers with scale factor 2
            predict = F.interpolate(predict, scale_factor=2, mode='trilinear')

    num_batches = predict.size(0)
    predict_flatten = predict.view(num_batches, -1)
    groundtruth_flatten = groundtruth.view(num_batches, -1)
    predict_flatten_sofmax = F.softmax(predict_flatten, dim=1)
    groundtruth_flatten_softmax = F.softmax(groundtruth_flatten, dim=1)

    return F.mse_loss(predict_flatten_sofmax, groundtruth_flatten_softmax)



#---------------------------------------------------------------------------------------------------
def focal_loss(predict, groundtruth, alpha=0.25, gamma=2.0):
    r'''
    Calculate the Focal loss

    : param alpha: balancing the positive and negative samples, by default 0.25
    : param gamma: penalizing the wrong predictions, default value 2
    '''
    clamped_predict = torch.clamp(predict, min=epsilon, max=(1.0 - epsilon))
    predict_flatten = clamped_predict.view(-1).float()
    detached_groundtruth = groundtruth.detach()
    groundtruth_flatten = detached_groundtruth.view(-1).float()

    indices = (groundtruth_flatten > 0)
    groundtruth_flatten_positive = groundtruth_flatten[indices]
    groundtruth_flatten_negative = groundtruth_flatten[~indices]
    predict_flatten_positive = predict_flatten[indices]
    predict_flatten_negative = predict_flatten[~indices]

    loss = 0
    if (predict_flatten_positive.size(0) != 0) and (groundtruth_flatten_positive.size(0) != 0):
        # positive samples
        log_pos = torch.log(predict_flatten_positive)
        loss += -1.0 * torch.mean(torch.pow(1.0 - predict_flatten_positive, gamma) * log_pos) * alpha

    if (predict_flatten_negative.size(0) != 0) and (groundtruth_flatten_negative.size(0) != 0):
        # negative samples
        log_neg = torch.log(1.0 - predict_flatten_negative)
        loss += -1.0 * torch.mean(torch.pow(predict_flatten_negative, gamma) * log_neg) * (1.0 - alpha)

    return loss


#---------------------------------------------------------------------------------------------------
def binary_cross_entropy_loss(predict, groundtruth):
    groundtruth_flatten = groundtruth.view(-1).float()
    predict_flatten = predict.view(-1).float()
    return F.binary_cross_entropy(predict_flatten, groundtruth_flatten)