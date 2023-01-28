#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : unet3d_with_feature_recalibration.py
#  Brief    : Based on the 3D UNet architecture, add the feature recalibration (FR) module.
#
#

import torch
import torch.nn as nn

from unet3d import UNet3D


# Functions ========================================================================================


# Classes ==========================================================================================
class UNet3DWithFeatureRecalibration(UNet3D):
    r'''
    UNet3D model with Feature Recalibration module for pulmonary airway segmentation.
    '''
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 Depth_max=80,
                 Height_max=192,
                 Width_max=304):
        r'''
        Parameters
        ----------
        in_channels     : the number of input channels
        out_channels    : the number of output channels
        Depth_max       : the size of largest feature recalibration cube in depth, default value is 80.
        Height_max      : the size of largest feature recalibration cube in height, 192 by default.
        Width_max       : the size of largest feature recalibration cube in width, 304 by default.
        '''
        super().__init__(in_channels, out_channels)
        # In the down-sampling path
        # arguments list:            num_channels  Depth              Height              Width
        self.FR1 = FeatureRecalibrationModule( 16, Depth_max,         Height_max,         Width_max)
        self.FR2 = FeatureRecalibrationModule( 32, (Depth_max //  2), (Height_max //  2), (Width_max //  2))
        self.FR3 = FeatureRecalibrationModule( 64, (Depth_max //  4), (Height_max //  4), (Width_max //  4))
        self.FR4 = FeatureRecalibrationModule(128, (Depth_max //  8), (Height_max //  8), (Width_max //  8))
        # In the bottom
        self.FR5 = FeatureRecalibrationModule(256, (Depth_max // 16), (Height_max // 16), (Width_max // 16))
        # In the up-sampling path
        self.FR6 = FeatureRecalibrationModule(128, (Depth_max //  8), (Height_max //  8), (Width_max //  8))
        self.FR7 = FeatureRecalibrationModule( 64, (Depth_max //  4), (Height_max //  4), (Width_max //  4))
        self.FR8 = FeatureRecalibrationModule( 32, (Depth_max //  2), (Height_max //  2), (Width_max //  2))
        self.FR9 = FeatureRecalibrationModule( 16, Depth_max,         Height_max,         Width_max)

    def forward(self, input_tensor):
        r'''
        :param input_tensor: the activated feature of m-th layer,
                             shape = [batch, channel, depth, height, width]
        :return: the segmented tensor, attention mapping list
        '''
        conv1 = self.conv1(input_tensor)
        recalibrated_feat1, feat_mapping1 = self.FR1(conv1)
        pooling1 = self.pooling(recalibrated_feat1)

        conv2 = self.conv2(pooling1)
        recalibrated_feat2, feat_mapping2 = self.FR2(conv2)
        pooling2 = self.pooling(recalibrated_feat2)

        conv3 = self.conv3(pooling2)
        recalibrated_feat3, feat_mapping3 = self.FR3(conv3)
        pooling3 = self.pooling(recalibrated_feat3)

        conv4 = self.conv4(pooling3)
        recalibrated_feat4, feat_mapping4 = self.FR4(conv4)
        pooling4 = self.pooling(recalibrated_feat4)

        conv5 = self.conv5(pooling4)
        recalibrated_feat5, feat_mapping5 = self.FR5(conv5)
        upsample5 = self.upsampling(recalibrated_feat5)

        concatenate_upsample5_conv4 = torch.cat([upsample5, conv4], dim=1)
        conv6 = self.conv6(concatenate_upsample5_conv4)
        recalibrated_feat6, feat_mapping6 = self.FR6(conv6)
        upsample6 = self.upsampling(recalibrated_feat6)

        concatenate_upsample6_conv3 = torch.cat([upsample6, conv3], dim=1)
        conv7 = self.conv7(concatenate_upsample6_conv3)
        recalibrated_feat7, feat_mapping7 = self.FR7(conv7)
        upsample7 = self.upsampling(recalibrated_feat7)

        concatenate_upsample7_conv2 = torch.cat([upsample7, conv2], dim=1)
        conv8 = self.conv8(concatenate_upsample7_conv2)
        recalibrated_feat8, feat_mapping8 = self.FR8(conv8)
        upsample8 = self.upsampling(recalibrated_feat8)

        concatenate_upsample8_conv1 = torch.cat([upsample8, conv1], dim=1)
        conv9 = self.conv9(concatenate_upsample8_conv1)
        recalibrated_feat9, feat_mapping9 = self.FR9(conv9)

        conv10 = self.conv10(recalibrated_feat9)
        output_seg_tensor = self.final_sigmoid(conv10)

        return  output_seg_tensor, \
                [feat_mapping3, feat_mapping4, feat_mapping5, feat_mapping6, feat_mapping7, feat_mapping8, feat_mapping9]


#---------------------------------------------------------------------------------------------------
class FeatureRecalibrationModule(nn.Module):
    def __init__(self, num_channels, Depth, Height, Width, reduction_ratio=2):
        r'''
        Parameters
        ----------
        num_channels            : number of the input channels
        Depth, Height, Width    : spatial dimension of the input feature cube
        reduction_ratio         : by how much should the num_channels be reduced
        '''
        super().__init__()
        num_reduced_channels = num_channels // reduction_ratio

        self.reduction_ratio = reduction_ratio
        self.conv_module = nn.Sequential(
            nn.Conv3d(in_channels=num_channels, out_channels=num_reduced_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=num_reduced_channels, out_channels=num_channels, kernel_size=1, stride=1),
            nn.Sigmoid())
        self.spatial_dimension = [Depth, Height, Width]
        self.Depth_squeeze  = nn.Conv3d(in_channels=Depth,  out_channels=1, kernel_size=1, stride=1)
        self.Height_squeeze = nn.Conv3d(in_channels=Height, out_channels=1, kernel_size=1, stride=1)
        self.Width_squeeze  = nn.Conv3d(in_channels=Width,  out_channels=1, kernel_size=1, stride=1)

    def forward(self, input_tensor):
        r'''
        Parameters
        ----------
        input_tensor    : input_tensor = Am, namely the activated feature of the m-th convolution layer.
                          the input_tensor.shape = [Batch, Channel, Depth, Height, Width]

        Returns
        -------
        output_tensor and feature_recalibration mapping
        '''
        squared_tensor = torch.pow(input_tensor, exponent=2)

        # Weight along channels and different axes
        Depth, Height, Width = self.spatial_dimension[0], self.spatial_dimension[1], self.spatial_dimension[2]
        Depth_axis = input_tensor.permute(0, 2, 1, 3, 4)        # Batch, Depth,  Channel, Height,  Width
        Height_axis = input_tensor.permute(0, 3, 2, 1, 4)       # Batch, Height, Depth,   Channel, Width

        # Step1: spatial map that highlights important regions is integrated through Z_spatial(*) along
        #        3 axes of depth, height and width.
        Z_spatial_integration_on_Depth = self.Height_squeeze(Height_axis).permute(0, 4, 2, 1, 3)
        Z_spatial_integration_on_Depth = self.Width_squeeze(Z_spatial_integration_on_Depth).permute(0, 4, 2, 3, 1)

        Z_spatial_integration_on_Height = self.Depth_squeeze(Depth_axis).permute(0, 4, 1, 3, 2)
        Z_spatial_integration_on_Height = self.Width_squeeze(Z_spatial_integration_on_Height).permute(0, 4, 2, 3, 1)

        Z_spatial_integration_on_Width = self.Depth_squeeze(Depth_axis).permute(0, 3, 1, 2, 4)
        Z_spatial_integration_on_Width = self.Height_squeeze(Z_spatial_integration_on_Width).permute(0, 3, 2, 1, 4)

        Z_spatial_integration = Z_spatial_integration_on_Depth + \
                                Z_spatial_integration_on_Height + \
                                Z_spatial_integration_on_Width

        # Step2: Channel recombination is performed on the spatial map to compute the channel descriptor Um
        channel_descriptor = self.conv_module(Z_spatial_integration)

        # The final element-wise multiplication between Am and Um produces the recalibrated feature
        recalibrated_feature = torch.mul(input_tensor, channel_descriptor)

        feature_mapping = torch.sum(squared_tensor, dim=1, keepdim=True)

        return recalibrated_feature, feature_mapping


#===================================================================================================
if __name__ == "__main__":
    net = UNet3DWithFeatureRecalibration(in_channels=1, out_channels=1)
    print(net)

    num_params = sum(param.numel() for param in net.parameters())
    print("Parameters Number of UNet3D + Feature_Recalibration network: {0}".format(num_params))

    # Remark here:
    # Parameters Number of UNet3D + Feature_Recalibration network: 4230352