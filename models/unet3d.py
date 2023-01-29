#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : unet3d.py
# Brief     : Design the 3D UNet architecture
#
#


import torch
import torch.nn as nn


# Functions ========================================================================================



# Classes ==========================================================================================
class UNet3D(nn.Module):
    r'''
    The 3D UNet model for pulmonary airway 3D segmentation
    '''
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=self._in_channels, out_channels=8,
                      kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.InstanceNorm3d(num_features=8, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=16, momentum=0.1, affine=False),
            nn.ReLU(inplace=True))

        self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=16, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=32, momentum=0.1, affine=False),
            nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=32, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=64, momentum=0.1, affine=False),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=64, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=128, momentum=0.1, affine=False),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=128, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=256, momentum=0.1, affine=False),
            nn.ReLU(inplace=True))

        self.upsampling = nn.Upsample(scale_factor=2.0, mode='nearest')

        self.conv6 = nn.Sequential(
            nn.Conv3d(in_channels=256 + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=128, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=128, momentum=0.1, affine=False),
            nn.ReLU(inplace=True))

        self.conv7 = nn.Sequential(
            nn.Conv3d(in_channels=128 + 64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=64, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(
            nn.Conv3d(in_channels=64 + 32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=32, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.Conv3d(in_channels=32 + 16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=16, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(num_features=16, momentum=0.1, affine=False),
            nn.ReLU(inplace=True))

        self.conv10 = nn.Conv3d(in_channels=16, out_channels=self._out_channels,
                                kernel_size=3, stride=1, padding=1)

        self.final_sigmoid = nn.Sigmoid()

    #-----------------------------------------------------------------------------------------------
    def forward(self, input_tensor):
        r'''
        Parameters
        ----------
        input_tensor : shape = torch.Size([batch_size, num_channels, Depth, Height, Width])

        Returns
        -------
        output segmentation tensor, attention mapping
        '''
        conv1 = self.conv1(input_tensor)
        pooling1 = self.pooling(conv1)

        conv2 = self.conv2(pooling1)
        pooling2 = self.pooling(conv2)

        conv3 = self.conv3(pooling2)
        pooling3 = self.pooling(conv3)

        conv4 = self.conv4(pooling3)
        pooling4 = self.pooling(conv4)

        conv5 = self.conv5(pooling4)

        upsample5 = self.upsampling(conv5)
        cat_upsample5_conv4 = torch.cat([upsample5, conv4], dim=1)
        conv6 = self.conv6(cat_upsample5_conv4)

        upsample6 = self.upsampling(conv6)
        cat_upsample6_conv3 = torch.cat([upsample6, conv3], dim=1)
        conv7 = self.conv7(cat_upsample6_conv3)

        upsample7 = self.upsampling(conv7)
        cat_upsample7_conv2 = torch.cat([upsample7, conv2], dim=1)
        conv8 = self.conv8(cat_upsample7_conv2)

        upsample8 = self.upsampling(conv8)
        cat_upsample8_conv1 = torch.cat([upsample8, conv1], dim=1)
        conv9 = self.conv9(cat_upsample8_conv1)

        conv10 = self.conv10(conv9)
        output_seg_tensor = self.final_sigmoid(conv10)

        # Calculate the attention distillation mapping
        ad_mapping3 = torch.sum(torch.pow(conv3, exponent=2), dim=1, keepdim=True)
        ad_mapping4 = torch.sum(torch.pow(conv4, exponent=2), dim=1, keepdim=True)
        ad_mapping5 = torch.sum(torch.pow(conv5, exponent=2), dim=1, keepdim=True)
        ad_mapping6 = torch.sum(torch.pow(conv6, exponent=2), dim=1, keepdim=True)
        ad_mapping7 = torch.sum(torch.pow(conv7, exponent=2), dim=1, keepdim=True)
        ad_mapping8 = torch.sum(torch.pow(conv8, exponent=2), dim=1, keepdim=True)
        ad_mapping9 = torch.sum(torch.pow(conv9, exponent=2), dim=1, keepdim=True)

        return output_seg_tensor, \
               [ad_mapping3, ad_mapping4, ad_mapping5, ad_mapping6, ad_mapping7, ad_mapping8, ad_mapping9]


#===================================================================================================
if __name__ == "__main__":
    net = UNet3D(in_channels=1, out_channels=1)
    print(net)

    num_parameters = sum(parameter.numel() for parameter in net.parameters())
    print("Parameters Number of UNet3D network: {0}".format(num_parameters))

    # Remark here
    # Parameters Number of UNet3D network: 4117969