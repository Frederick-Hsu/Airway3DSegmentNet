#!/user/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : splitter_combiner.py
# Brief     : Design a class "SplitterCombiner" to guide how to split the CT 3D images to many
#             cubes, then combine these cubes.
#
#

import numpy as np


# Classes ==========================================================================================

class SplitterCombiner:
    def __init__(self, crop_cube_size, crop_stride):
        r'''
        Split a CT 3D image into many sub-cubes, initialize the parameters of cropped cube size and
        crop stride.
        Parameters
        ----------
        crop_cube_size : specify the cube size [Depth, Height, Width] of each sub-cube.
        crop_stride : the sliding stride [Depth, Height, Width] you want to crop along
        the depth/height/width 3 directions.
        '''
        assert isinstance(crop_cube_size, (int, tuple, list)), \
            "Error: the crop_cube_size must be 3-elem tuple or list"
        if type(crop_cube_size) is int:
            crop_cube_size = [crop_cube_size, crop_cube_size, crop_cube_size]
        else:
            assert len(crop_cube_size) == 3

        assert isinstance(crop_stride, (int, tuple, list))
        if type(crop_stride) is int:
            crop_stride = [crop_stride, crop_stride, crop_stride]
        else:
            assert len(crop_stride) == 3


        self.cubesize = crop_cube_size  # shape: [Depth, Height, Width]
        self.stride = crop_stride       # shape: [D, H, W]

    def split(self, CT_3d_image):
        r'''
        Split the CT 3D image into sub-cubes, according to the specified cube size and stride
        Parameters
        ----------
        CT_3d_image : the raw CT 3D image, its shape = [Depth, Height, Width]

        Returns
        -------
        output the list of coordinates for the cropped sub-cubes,
        from start-position to end-position
        '''
        cube_size = self.cubesize
        stride = self.stride

        splits = []
        depth, height, width = CT_3d_image.shape

        num_depth   = int(np.ceil(float(depth  - cube_size[0])/stride[0]))
        num_height  = int(np.ceil(float(height - cube_size[1])/stride[1]))
        num_width   = int(np.ceil(float(width  - cube_size[2])/stride[2]))

        assert (num_depth  * stride[0] + cube_size[0] - depth) >= 0
        assert (num_height * stride[1] + cube_size[1] - height) >= 0
        assert (num_width  * stride[2] + cube_size[2] - width) >= 0

        num_DHW = [num_depth, num_height, num_width]
        self.num_DHW = num_DHW

        padding = [[0, num_depth  * stride[0] + cube_size[0] - depth ],
                   [0, num_height * stride[1] + cube_size[1] - height],
                   [0, num_width  * stride[2] + cube_size[2] - width ]]

        raw_CT_3d_image_shape = [depth, height, width]

        count = 0
        for index_depth in range(num_depth + 1):
            for index_height in range(num_height + 1):
                for index_width in range(num_width + 1):
                    # calculate the start index in depth direction
                    start_depth = index_depth * stride[0]
                    # calculate the end index in depth direction
                    end_depth   = index_depth * stride[0] + cube_size[0]

                    # calculate the start and end index in height direction
                    start_height = index_height * stride[1]
                    end_height   = index_height * stride[1] + cube_size[1]

                    # calculate the start and end index in width direction
                    start_width  = index_width * stride[2]
                    end_width    = index_width * stride[2] + cube_size[2]

                    if end_depth > depth:
                        start_depth = depth - cube_size[0]
                        end_depth   = depth
                    if end_height > height:
                        start_height = height - cube_size[1]
                        end_height   = height
                    if end_width > width:
                        start_width  = width - cube_size[2]
                        end_width    = width

                    cube_crop_index = [[start_depth,  end_depth ],
                                       [start_height, end_height],
                                       [start_width,  end_width ],
                                       count]
                    splits.append(cube_crop_index)
                    count += 1

        splits = np.array(splits, dtype=type(splits))
        return splits, num_DHW, raw_CT_3d_image_shape


#===================================================================================================
if __name__ == "__main__":
    from utils import load_CT_scan_3D_image

    ct_image_file_path = "../LearningTubuleSensitiveCNNs/preprocessed_datasets/imagesTr/ATM_127_0000_clean_hu.nii.gz"
    image_np, origin, spacing = load_CT_scan_3D_image(ct_image_file_path)

    splitter = SplitterCombiner(crop_stride=[64, 96, 152], crop_cube_size=[80, 192, 304])
    # splitter = SplitComb(side_len=[64, 96, 152], margin=[80, 192, 304])
    splits, num_DHW, shape = splitter.split(image_np)
    print(splits)
    print(num_DHW)
    print(shape)