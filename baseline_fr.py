#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : baseline_fr.py
# Brief     : Retrieve the model of baseline + feature recalibration
#
#


from models.unet3d_with_feature_recalibration import UNet3DWithFeatureRecalibration
from baseline import config


# Functions ========================================================================================
def get_model(args=None):
    net = UNet3DWithFeatureRecalibration(in_channels=1, out_channels=1,
                                         Depth_max=args.train_cube_size[0],
                                         Height_max=args.train_cube_size[1],
                                         Width_max=args.train_cube_size[2])
    print(net)
    num_parameters = sum(param.numel() for param in net.parameters())
    print("Number of network parameters: {0}".format(num_parameters))
    return config, net