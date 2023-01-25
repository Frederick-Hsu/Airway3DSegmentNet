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
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 Depth_max=80,
                 Height_max=192,
                 Width_max=304):
        super().__init__(in_channels, out_channels)

    def forward(self, input_tensor):
        pass