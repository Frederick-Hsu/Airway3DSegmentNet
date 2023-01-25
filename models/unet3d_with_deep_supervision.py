#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : unet3d_with_deep_supervision.py
# Brief     : Based on the 3D UNet architecture, add the deep supervision (DS) module
#
#

from unet3d import UNet3D

# Functions ========================================================================================


# Classes ==========================================================================================
class UNet3DWithDeepSupervision(UNet3D):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__(in_channels, out_channels)

    def forward(self, input_tensor):
        pass

#===================================================================================================
if __name__ == "__main__":
    pass