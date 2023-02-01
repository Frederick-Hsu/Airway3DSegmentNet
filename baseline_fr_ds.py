#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : baseline_fr_ds.py
# Brief     : Get the model of UNet3D + Feature_Recalibration + Deep_Supervision network
#
#


from models.unet3d_with_FR_and_DS import UNet3DWithFeatureRecalibrationAndDeepSupervision
from baseline import config


# Functions ========================================================================================
def get_model(args=None):
    net = UNet3DWithFeatureRecalibrationAndDeepSupervision(in_channels=1,
                                                           out_channels=1,
                                                           Depth_max=args.train_cube_size[0],
                                                           Height_max=args.train_cube_size[1],
                                                           Width_max=args.train_cube_size[2])
    print(net)
    num_parameters = sum(param.numel() for param in net.parameters())
    print("Parameters Number of UNet3D + Feature_Recalibration + Deep_Supervision network: {0}"
          .format(num_parameters))
    return config, net