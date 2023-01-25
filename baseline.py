#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : baseline.py
# Brief     : Retrieve the baseline model.
#
#

import numpy as np

from models.unet3d import UNet3D

# Objects ==========================================================================================
config = {'augtype': {'flip': True,
                      'swap': False,
                      'smooth': False,
                      'jitter': True,
                      'split_jitter': True},
          'lr_stage': np.array([   10,    20,    40,    60]),
          'lr':       np.array([3e-03, 3e-04, 3e-05, 3e-06]),
          'dataset_path' : "preprocessed_datasets",
          'dataset_split': "./split_dataset.pkl"}

# Functions ========================================================================================
def get_model(args=None):
    net = UNet3D(in_channels=1, out_channels=1)
    print(net)
    num_params = sum(param.numel() for param in net.parameters())
    print("Number of network parameters: {0}".format(num_params))
    return config, net



#===================================================================================================
if __name__ == "__main__":
    model_name = 'baseline'
    config, net = get_model(args=model_name)

