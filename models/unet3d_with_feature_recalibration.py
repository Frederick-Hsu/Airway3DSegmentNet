#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : unet3d_with_feature_recalibration.py
#  Brief    : Based on the 3D UNet architecture, add the feature recalibration (FR) module.
#
#


from .unet3d import UNet3D


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
        pass
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

