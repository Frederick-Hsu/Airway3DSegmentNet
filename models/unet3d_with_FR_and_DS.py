#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : unet3d_with_FR_and_DS.py
# Brief     : Design a neural network: UNet3D + Feature Recalibration + Deep Supervision 2 add-on modules
#
#


import torch
import torch.nn as nn

from .unet3d_with_feature_recalibration import UNet3DWithFeatureRecalibration


# Classes ==========================================================================================
class UNet3DWithFeatureRecalibrationAndDeepSupervision(UNet3DWithFeatureRecalibration):
    r'''
    UNet3D + Feature_Recalibration + Deep_Supervision model for pulmonary airway 3D segmentation
    '''
    def __init__(self, in_channels=1, out_channels=1, Depth_max=80, Height_max=192, Width_max=304):
        super().__init__(in_channels, out_channels, Depth_max, Height_max, Width_max)

        self.upsampling2 = nn.Upsample(scale_factor=2)
        self.upsampling4 = nn.Upsample(scale_factor=4)
        self.upsampling8 = nn.Upsample(scale_factor=8)

        self.deep_supervision_conv6 = nn.Conv3d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.deep_supervision_conv7 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.deep_supervision_conv8 = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        # recal_feat: recalibrated feature,  ad_mapping: attention distillation feature mapping
        recal_feat1, ad_mapping1 = self.FR1(conv1)
        pooling1 = self.pooling(recal_feat1)

        conv2 = self.conv2(pooling1)
        recal_feat2, ad_mapping2 = self.FR2(conv2)
        pooling2 = self.pooling(recal_feat2)

        conv3 = self.conv3(pooling2)
        recal_feat3, ad_mapping3 = self.FR3(conv3)
        pooling3 = self.pooling(recal_feat3)

        conv4 = self.conv4(pooling3)
        recal_feat4, ad_mapping4 = self.FR4(conv4)
        pooling4 = self.pooling(recal_feat4)

        conv5 = self.conv5(pooling4)
        recal_feat5, ad_mapping5 = self.FR5(conv5)

        upsample5 = self.upsampling(recal_feat5)
        concat_upsample5_recalfeat4 = torch.cat([upsample5, recal_feat4], dim=1)
        conv6 = self.conv6(concat_upsample5_recalfeat4)
        recal_feat6, ad_mapping6 = self.FR6(conv6)
        deep_supervision6 = self.final_sigmoid(self.upsampling8(self.deep_supervision_conv6(recal_feat6)))

        upsample6 = self.upsampling(recal_feat6)
        concat_upsample6_recalfeat3 = torch.cat([upsample6, recal_feat3], dim=1)
        conv7 = self.conv7(concat_upsample6_recalfeat3)
        recal_feat7, ad_mapping7 = self.FR7(conv7)
        deep_supervision7 = self.final_sigmoid(self.upsampling4(self.deep_supervision_conv7(recal_feat7)))

        upsample7 = self.upsampling(recal_feat7)
        concat_upsample7_recalfeat2 = torch.cat([upsample7, recal_feat2], dim=1)
        conv8 = self.conv8(concat_upsample7_recalfeat2)
        recal_feat8, ad_mapping8 = self.FR8(conv8)
        deep_supervision8 = self.final_sigmoid(self.upsampling2(self.deep_supervision_conv8(recal_feat8)))

        upsample8 = self.upsampling(recal_feat8)
        concat_upsample8_recalfeat1 = torch.cat([upsample8, recal_feat1], dim=1)
        conv9 = self.conv9(concat_upsample8_recalfeat1)
        recal_feat9, ad_mapping9 = self.FR9(conv9)

        conv10 = self.conv10(recal_feat9)
        output_seg_tensor = self.final_sigmoid(conv10)

        return  [output_seg_tensor, deep_supervision6, deep_supervision7, deep_supervision8], \
                [ad_mapping3, ad_mapping4, ad_mapping5, ad_mapping6, ad_mapping7, ad_mapping8, ad_mapping9]


#===================================================================================================
if __name__ == "__main__":
    net = UNet3DWithFeatureRecalibrationAndDeepSupervision(in_channels=1, out_channels=1)
    print(net)

    params_num = sum(param.numel() for param in net.parameters())
    print("Parameters Number of UNet3D + Feature_Recalibration + Deep_Supervision network: {0}"
          .format(params_num))

    # Remark here:
    # Parameters Number of UNet3D + Feature_Recalibration + Deep_Supervision network: 423,6403
