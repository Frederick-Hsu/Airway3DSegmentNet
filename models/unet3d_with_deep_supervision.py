#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : unet3d_with_deep_supervision.py
# Brief     : Based on the 3D UNet architecture, add the deep supervision (DS) module
#
#


import torch
import torch.nn as nn

from unet3d import UNet3D


# Functions ========================================================================================


# Classes ==========================================================================================
class UNet3DWithDeepSupervision(UNet3D):
    r'''
    UNet3D + Deep_Supervision model for pulmonary 3D airway segmentation
    '''
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(in_channels, out_channels)

        self.upsampling2 = nn.Upsample(scale_factor=2)
        self.upsampling4 = nn.Upsample(scale_factor=4)
        self.upsampling8 = nn.Upsample(scale_factor=8)

        self.deep_supervision_conv6 = nn.Conv3d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.deep_supervision_conv7 = nn.Conv3d(in_channels=64,  out_channels=1, kernel_size=3, stride=1, padding=1)
        self.deep_supervision_conv8 = nn.Conv3d(in_channels=32,  out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        pooling1 = self.pooling(conv1)

        conv2 = self.conv2(pooling1)
        pooling2 = self.pooling(conv2)

        conv3 = self.conv3(pooling2)
        pooling3 = self.pooling(conv3)

        conv4 = self.conv4(pooling3)
        pooling4 = self.pooling(conv4)

        conv5 = self.conv5(pooling4)

        upsampling5 = self.upsampling(conv5)
        concatenate_upsampling5_conv4 = torch.cat([upsampling5, conv4], dim=1)
        conv6 = self.conv6(concatenate_upsampling5_conv4)

        deep_supervision6 = self.final_sigmoid(self.upsampling8(self.deep_supervision_conv6(conv6)))

        upsampling6 = self.upsampling(conv6)
        concatenate_upsampling6_conv3 = torch.cat([upsampling6, conv3], dim=1)
        conv7 = self.conv7(concatenate_upsampling6_conv3)

        deep_supervision7 = self.final_sigmoid(self.upsampling4(self.deep_supervision_conv7(conv7)))

        upsampling7 = self.upsampling(conv7)
        concatenate_upsampling7_conv2 = torch.cat([upsampling7, conv2], dim=1)
        conv8 = self.conv8(concatenate_upsampling7_conv2)

        deep_supervision8 = self.final_sigmoid(self.upsampling2(self.deep_supervision_conv8(conv8)))

        upsampling8 = self.upsampling(conv8)
        concatenate_upsampling8_conv1 = torch.cat([upsampling8, conv1], dim=1)
        conv9 = self.conv9(concatenate_upsampling8_conv1)

        conv10 = self.conv10(conv9)

        output_seg_tensor = self.final_sigmoid(conv10)

        ad_mapping3 = torch.sum(torch.pow(conv3, exponent=2), dim=1, keepdim=True)
        ad_mapping4 = torch.sum(torch.pow(conv4, exponent=2), dim=1, keepdim=True)
        ad_mapping5 = torch.sum(torch.pow(conv5, exponent=2), dim=1, keepdim=True)
        ad_mapping6 = torch.sum(torch.pow(conv6, exponent=2), dim=1, keepdim=True)
        ad_mapping7 = torch.sum(torch.pow(conv7, exponent=2), dim=1, keepdim=True)
        ad_mapping8 = torch.sum(torch.pow(conv8, exponent=2), dim=1, keepdim=True)
        ad_mapping9 = torch.sum(torch.pow(conv9, exponent=2), dim=1, keepdim=True)

        return  [output_seg_tensor, deep_supervision6, deep_supervision7, deep_supervision8], \
                [ad_mapping3, ad_mapping4, ad_mapping5, ad_mapping6, ad_mapping7, ad_mapping8, ad_mapping9]


#===================================================================================================
if __name__ == "__main__":
    pass