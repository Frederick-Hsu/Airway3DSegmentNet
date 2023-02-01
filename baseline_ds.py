#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : baseline_ds.py
# Brief     : Get the baseline + deep_supervision model
#
#


from models.unet3d_with_deep_supervision import UNet3DWithDeepSupervision
from baseline import config


# Functions ========================================================================================
def get_model(args=None):
    net = UNet3DWithDeepSupervision(in_channels=1, out_channels=1)
    print(net)

    num_params = sum(param.numel() for param in net.parameters())
    print("Parameters Number of UNet3D + Deep_Supervision network: {0}".format(num_params))
    return config, net


#===================================================================================================
if __name__ == "__main__":
    model_name = "baseline_ds"
    curr_config, curr_net = get_model(args=model_name)