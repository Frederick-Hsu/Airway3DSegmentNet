#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File  : segment_3d_airway_model.py
#
#


import torch
import torch.nn as nn
from functools import partial

# Functions ========================================================================================
def create_conv(in_channels, 
                out_channels, 
                kernel_size, 
                order, 
                num_groups, 
                padding, 
                stride, 
                dilation=1):
    r'''
    Create a list of modules with together, constitute a single convolution layer with 
    non-linearity and optional BatchNorm/GroupNorm.

    Parameters
    ----------
    in_channels : int
        number of input channels
        
    out_channels : int
        number of output channels
        
    kernel_size : int or tuple
        size of convolution kernel
        
    order : str
        order of things, e.g.
        'cr'  ---> Conv + ReLU
        'gcr' ---> GroupNorm + Conv + ReLU
        'cl'  ---> Conv + LeakyReLU
        'ce'  ---> Conv + ELU
        'bcr' ---> BatchNorm + Conv + ReLU
        
    num_groups : int
        number of groups for the GroupNorm
        
    padding : int or tuple
        add zero-padding, added to all 3 sides of the input
        
    Returns
    -------
    list of tuple (name, module)
    '''
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', "Non-linearity cannot be the first operation in the layer."
    
    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of BatchNorm/GroupNorm
            bias = not (('g' in order) or ('b' in order))
            modules.append(('conv', nn.Conv3d(in_channels, 
                                              out_channels, 
                                              kernel_size, 
                                              padding=padding,
                                              bias=bias,
                                              stride=stride,
                                              dilation=dilation)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels
            # use only one group, if the given numbers is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1
            
            assert num_channels % num_groups == 0, \
                ("Expected number of channels in input to be divisible by num_groups. "
                 "num_channels={0}, num_groups={1}".format(num_channels, num_groups))
            
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, 
                                                      num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError("Unsupported layer type '{0}'."
                             "MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']".format(char))
    
    return modules


# Classes ==========================================================================================
class SingleConv(nn.Sequential):
    r'''
    Basic convolutional module consists of a Conv3d, non-linearity and optional BatchNorm/GroupNorm.
    The order of operations can be specified via the 'order' parameter.
    '''
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 order='gcr', 
                 num_groups=8,
                 padding=1,
                 stride=1,
                 dilation=1):
        r'''
        Parameters
        ----------
        in_channels : int
            number of input channels
            
        out_channels : int
            number of output channels
            
        kernel_size : int or tuple
            size of convolution kernel
            
        order : str
            determines the order of layers, e.g.
            'cr'  ---> Conv + ReLU
            'crg' ---> Conv + ReLU + GroupNorm
            'cl'  ---> Conv + LeakyReLU
            'ce'  ---> Conv + ELU
            
        num_groups : int
            number of groups for the GroupNorm
            
        padding : int or tuple
        '''
        super(SingleConv, self).__init__()
        
        for name, module in create_conv(in_channels, 
                                        out_channels, 
                                        kernel_size, 
                                        order, 
                                        num_groups,
                                        padding=padding,
                                        stride=stride,
                                        dilation=dilation):
            self.add_module(name, module)

#---------------------------------------------------------------------------------------------------
class AttentionModule(nn.Module):
    def __init__(self, channel, mid_channel=8):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(channel, mid_channel, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(mid_channel, channel, bias=False),
                                nn.Sigmoid())
    
    def forward(self, x):
        batch, channel, _, _, _ = x.size()
        y = self.avg_pool(x).view(batch, channel)
        y = self.fc(y).view(batch, channel, 1, 1, 1)
        return x * y.expand_as(x)

#---------------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    r'''
    A single encoder module consisting of
    
    (1) two consecutive convolution layers  (e.g.  BatchNorm3d + ReLU + Conv3d)
        We use (Conv3d + ReLU + GroupNorm3d) by default.
        This can be changed however by providing the `order` argument, e.g. in order to change to
        (Conv3d + BatchNorm3d + ELU) use order="cbe".
        Use padded convolutions to make sure that the output (H_out, W_out) is the same as 
        (H_in, W_in), so that you don't have to crop in the decoder path.
    
    (2) the pooling layer
        one may specify the MaxPool kernel size to be different than the standard (2, 2, 2),
        e.g. if the volumetric data is anisotropic followed by a DoubleConv module, to make sure to
        use complementary scale_factor in the decoder path.
    '''
    def __init__(self,
                 in_channels : int,
                 middle_channels : int,
                 out_channels : int,
                 apply_pooling : bool = True,
                 conv_kernel_size : (int, tuple) = 3,
                 pool_kernel_size : (int, tuple) = 2,
                 pool_type : str = 'max',
                 conv_layer_order : str = 'gcr',
                 num_groups : int = 8,
                 padding : (int, tuple) = 1,
                 stride=1):
        r'''
        Parameters
        ----------
        in_channels : int
            number of input channels
            
        middle_channels : int
            number of middle channels
            
        out_channels : int
            number of output channels
            
        apply_pooling : bool
            if True use pooling
            
        conv_kernel_size : int or tuple
            size of the convolution kernel, default value is 3
            
        pool_kernel_size : int or tuple
            the size of pooling window, default value is 2
            
        pool_type : str
            pooling layer : "max" or "avg", default value is "max"
            
        conv_layer_order : str
            determines the order of layers, e.g.
            'cr'  ---> conv + ReLU
            'crg' ---> conv + ReLU + GroupNorm
            'cl'  ---> conv + LeakyReLU
            'ce'  ---> conv + ELU
            
        num_groups : int
            number of groups for the GroupNorm
            
        padding : int or tuple
            add zero-padding, added to all 3 sides of the input
        '''
        super().__init__()
        assert pool_type in ['max', 'avg']
        
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None
        
        # conv1
        self.conv1 = SingleConv(in_channels, 
                                middle_channels, 
                                conv_kernel_size, 
                                conv_layer_order,
                                num_groups,
                                padding=padding,
                                stride=stride)
        # conv2
        self.conv2 = SingleConv(middle_channels, 
                                out_channels, 
                                conv_kernel_size, 
                                conv_layer_order,
                                num_groups,
                                padding=padding,
                                stride=stride)
        # dilation conv
        self.dilation_conv = SingleConv(middle_channels, 
                                        middle_channels, 
                                        (3, 3, 3), 
                                        conv_layer_order, 
                                        num_groups,
                                        padding=(4, 4, 4),
                                        stride=stride,
                                        dilation=4)
        # attention mechanism
        self.attention = AttentionModule(channel=middle_channels)
    
    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.conv1(x)
        x = x + self.attention(self.dilation_conv(x))
        x = self.conv2(x)
        
        return x

#---------------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    r'''
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspond3d or nearest neighbour interpolation) followed by a basic module
    (e.g. double conv like encoder)
    '''
    def __init__(self, 
                 in_channels,
                 upsample_out_channels,
                 conv_in_channels,
                 conv_middle_channels,
                 out_channels,
                 conv_kernel_size=3,
                 conv_layer_order='gcr',
                 num_groups=8,
                 conv_padding=1,
                 conv_stride=1,
                 deconv_kernel_size=4,
                 deconv_stride=(2, 2, 2),
                 deconv_padding=1):
        super().__init__()
        # deconv
        self.upsample = nn.ConvTranspose3d(in_channels, 
                                           upsample_out_channels, 
                                           kernel_size=deconv_kernel_size, 
                                           stride=deconv_stride,
                                           padding=deconv_padding)
        # concat joining
        self.joining = partial(self._joining, concat=True)
        # conv1
        self.conv1 = SingleConv(conv_in_channels, 
                                conv_middle_channels, 
                                conv_kernel_size,
                                conv_layer_order,
                                num_groups,
                                padding=conv_padding,
                                stride=conv_stride)
        # conv2
        self.conv2 = SingleConv(conv_middle_channels,
                                out_channels,
                                conv_kernel_size,
                                conv_layer_order,
                                num_groups,
                                padding=conv_padding,
                                stride=conv_stride)
        # dilation conv
        self.dilation_conv = SingleConv(conv_middle_channels, 
                                        conv_middle_channels,
                                        (3, 3, 3),
                                        conv_layer_order,
                                        num_groups,
                                        padding=(4, 4, 4),
                                        stride=conv_stride,
                                        dilation=4)
        # attention layer
        self.attention = AttentionModule(channel=conv_middle_channels)
    
    def forward(self, encoder_features, x):
        x = self.upsample(x)
        x = self.joining(encoder_features, x)
        x = self.conv1(x)
        x = x + self.attention(self.dilation_conv(x))
        x = self.conv2(x)
        
        return x
    
    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x

#---------------------------------------------------------------------------------------------------
class SegAirwayModel(nn.Module):
    r'''
    The neural networks model for Pulmonary airway tree segmentation 
    '''
    def __init__(self, 
                 in_channels : int, 
                 out_channels : int, 
                 layer_order : str = "gcr", 
                 **kwargs):
        r'''
        Parameters
        ----------
        in_channels : int
            number of input channels
            
        out_channels : int
            number of output segmentation masks
            [NOTE] the `output_channels` might correspond to either different semantic classes
                   or to different binary segmentation mask.
                   It is up to the user of the class to interpret the `out_channels` and use the 
                   proper loss criterion during training (i.e. CrossEntrophyLoss (multi-class) or
                   BCEWithLogitsLoss (two-class) respectively.)
        
        layer_order : str, optional
            determines the order of layers in `SingleConv` module.
            e.g. 'crg' stands for Conv3d + ReLU + GroupNorm3d
            See `SingleConv` for more details.
            default value is "gcr"
        '''
        super(SegAirwayModel, self).__init__()
        
        # create encoder
        encoder_1 = Encoder(in_channels=in_channels,
                            middle_channels=16,
                            out_channels=32,
                            apply_pooling=False,
                            conv_kernel_size=3,
                            pool_kernel_size=2,
                            pool_type='max',
                            conv_layer_order=layer_order,
                            num_groups=8,
                            padding=1,
                            stride=1)
        encoder_2 = Encoder(in_channels=32,
                            middle_channels=32, 
                            out_channels=64, 
                            apply_pooling=True,
                            conv_kernel_size=3,
                            pool_kernel_size=2,
                            pool_type='max',
                            conv_layer_order=layer_order,
                            num_groups=8,
                            padding=1,
                            stride=1)
        encoder_3 = Encoder(in_channels=64, 
                            middle_channels=64,
                            out_channels=128,
                            apply_pooling=True,
                            conv_kernel_size=3,
                            pool_kernel_size=2,
                            pool_type='max',
                            conv_layer_order=layer_order,
                            num_groups=8,
                            padding=1,
                            stride=1)
        encoder_4 = Encoder(in_channels=128,
                            middle_channels=128,
                            out_channels=256,
                            apply_pooling=True,
                            conv_kernel_size=3,
                            pool_kernel_size=2,
                            pool_type='max',
                            conv_layer_order=layer_order,
                            num_groups=8,
                            padding=1,
                            stride=1)
        
        self.encoders = nn.ModuleList([encoder_1, encoder_2, encoder_3, encoder_4])
        
        # create decoder
        decoder_1 = Decoder(in_channels=256, 
                            upsample_out_channels=256, 
                            conv_in_channels=384,
                            conv_middle_channels=128,
                            out_channels=128,
                            conv_kernel_size=3,
                            conv_layer_order=layer_order,
                            num_groups=8,
                            conv_padding=1,
                            deconv_kernel_size=4,
                            deconv_stride=(2, 2, 2),
                            deconv_padding=1)
        decoder_2 = Decoder(in_channels=128,
                            upsample_out_channels=128,
                            conv_in_channels=192,
                            conv_middle_channels=64,
                            out_channels=64,
                            conv_kernel_size=3,
                            conv_layer_order=layer_order,
                            num_groups=8,
                            conv_padding=1,
                            conv_stride=1,
                            deconv_kernel_size=4,
                            deconv_stride=(2, 2, 2),
                            deconv_padding=1)
        decoder_3 = Decoder(in_channels=64,
                            upsample_out_channels=64,
                            conv_in_channels=96,
                            conv_middle_channels=32,
                            out_channels=32,
                            conv_kernel_size=3,
                            conv_layer_order=layer_order,
                            num_groups=8,
                            conv_padding=1,
                            conv_stride=1,
                            deconv_kernel_size=4,
                            deconv_stride=(2, 2, 2),
                            deconv_padding=1)
        
        self.decoders = nn.ModuleList([decoder_1, decoder_2, decoder_3])
        
        # in the last layer, a 1x1 convolution reduces the number of output channels to 
        # the number of labels
        self.fincal_conv = nn.Conv3d(in_channels=32, 
                                     out_channels=out_channels, 
                                     kernel_size=3, 
                                     padding=1,
                                     stride=1)
        self.final_activation = nn.Softmax(dim=1)
    
    def forward(self, x):
        # encoder path
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
        
        # remove the last encoder's output from the list
        encoders_features = encoders_features[1:]
        
        # decoder path
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder,
            # and the output of previous decoder
            x = decoder(encoder_features, x)
        
        x = self.fincal_conv(x)
        x = self.final_activation(x)
        
        return x

# Main business ====================================================================================
if __name__ == "__main__":
    pass