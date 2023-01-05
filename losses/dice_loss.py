#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File  : dice_loss.py
#
#
#


import torch

# Functions ========================================================================================
def dice_loss_weights(predict_values, target_values, weights):
    r'''
    This definition generalize to real values predict and target vector.
    It should be differentiable.

    Parameters
    ----------
    predict_values : torch.tensor
        tensor with first dimension as batch
        
    target_values : torch.tensor
        tensor with first dimension as batch
    '''
    smooth = 0.01
    
    # have to use contiguous since they may from a torch.view operation
    pred_flat = predict_values.contiguous().view(-1)
    target_flat = target_values.contiguous().view(-1)
    weight_flat = weights.contiguous().view(-1)
    
    intersection = 2.0 * torch.sum(torch.mul(torch.mul(pred_flat, target_flat), weight_flat))
    A_sum = torch.sum(torch.mul(pred_flat, pred_flat))
    B_sum = torch.sum(torch.mul(target_flat, target_flat))
    
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

#---------------------------------------------------------------------------------------------------
def dice_loss_power_weights(predict_values, target_values, weights, alpha=0.5, delta=0.1):
    smooth = 0.01
    
    pred_flat = predict_values.contiguous().view(-1)
    target_flat = target_values.contiguous().view(-1)
    weight_flat = weights.contiguous().view(-1)
    
    intersection = 2.0 * torch.sum(torch.mul(torch.mul(torch.pow(pred_flat + delta, alpha), 
                                                       target_flat), 
                                             weight_flat))
    
    A_sum = torch.sum(torch.mul(torch.mul(torch.pow(pred_flat + delta, alpha),
                                          torch.pow(pred_flat + delta, alpha)), 
                                weight_flat))
    
    B_sum = torch.sum(torch.mul(torch.mul(target_flat, target_flat), weight_flat))
    
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

#---------------------------------------------------------------------------------------------------
def dice_loss_power(predict_values, target_values, alpha=0.5, delta=0.1):
    smooth = 0.01
    
    pred_flat = predict_values.contiguous().view(-1)
    target_flat = target_values.contiguous().view(-1)
    
    intersection = 2.0 * torch.sum(torch.mul(torch.pow(pred_flat + delta, alpha), 
                                             target_flat))
    
    A_sum = torch.sum(torch.mul(torch.pow(pred_flat + delta, alpha),
                                torch.pow(pred_flat + delta, alpha)))
    
    B_sum = torch.sum(torch.mul(target_flat, target_flat))
    
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))
    
#---------------------------------------------------------------------------------------------------
def dice_loss(predict_values, target_values):
    smooth = 0.01
    
    pred_flat = predict_values.contiguous().view(-1)
    target_flat = target_values.contiguous().view(-1)
    
    intersection = 2.0 * torch.sum(torch.mul(pred_flat, target_flat))
    A_sum = torch.sum(torch.mul(pred_flat, pred_flat))
    B_sum = torch.sum(torch.mul(target_flat, target_flat))
    
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

#---------------------------------------------------------------------------------------------------
def dice_accuracy(predict_values, target_values):
    pred_flat = predict_values.contiguous().view(-1)
    target_flat = target_values.contiguous().view(-1)
    
    intersection = 2.0 * torch.sum(torch.mul(pred_flat, target_flat))
    
    A_sum = torch.sum(torch.mul(pred_flat, pred_flat))
    B_sum = torch.sum(torch.mul(target_flat, target_flat))
    
    return intersection / (A_sum + B_sum + 0.0001)


# Classes ==========================================================================================


# Tests ============================================================================================
if __name__ == "__main__":
    pass