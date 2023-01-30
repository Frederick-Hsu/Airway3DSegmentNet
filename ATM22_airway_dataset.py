#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : ATM22_airway_dataset.py
# Brief     : Organize the cropped CT 3D images and labels, according to the split_dataset.pkl file.
#
#


# System's modules
import os.path
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

# User-defined modules
from utils import load_pickle, save_pickle
from utils import load_CT_scan_3D_image
from splitter_combiner import SplitterCombiner
from log_switch import log


# Functions ========================================================================================
def augment_jittering_for_imagecube(split, cube_shape):
    # shape: [Depth, Height, Width]
    depth_start_index,  depth_end_index     = split[0][0], split[0][1]
    height_start_index, height_end_index    = split[1][0], split[1][1]
    width_start_index,  width_end_index     = split[2][0], split[2][1]

    curr_depth_jitter, curr_height_jitter, curr_width_jitter = 0, 0, 0
    if depth_end_index - depth_start_index <= 3:
        jitter_range = (depth_end_index - depth_start_index) * 32
    else:
        jitter_range = (depth_end_index - depth_start_index) * 2
    log.warning("\nJitter range = {0}".format(jitter_range))
    jitter_range_half = jitter_range // 2

    cnt = 0
    while cnt < 10:
        if depth_start_index == 0:
            curr_depth_jitter = int(np.random.rand() * jitter_range)
        elif depth_end_index == cube_shape[0]:
            curr_depth_jitter = -int(np.random.rand() * jitter_range)
        else:
            curr_depth_jitter = int(np.random.rand() * jitter_range) - jitter_range_half
        cnt += 1
        if  (curr_depth_jitter + depth_start_index >= 0) and \
            (curr_depth_jitter + depth_end_index < cube_shape[0]):
            break

    cnt = 0
    while cnt < 10:
        if height_start_index == 0:
            curr_height_jitter = int(np.random.rand() * jitter_range)
        elif height_end_index == cube_shape[1]:
            curr_height_jitter = -int(np.random.rand() * jitter_range)
        else:
            curr_height_jitter = int(np.random.rand() * jitter_range) - jitter_range_half
        cnt += 1
        if  (curr_height_jitter + height_start_index >= 0) and \
            (curr_height_jitter + height_end_index < cube_shape[1]):
            break

    cnt = 0
    while cnt < 10:
        if width_start_index == 0:
            curr_width_jitter = int(np.random.rand() * jitter_range)
        elif width_end_index == cube_shape[2]:
            curr_width_jitter = -int(np.random.rand() * jitter_range)
        else:
            curr_width_jitter = int(np.random.rand() * jitter_range) - jitter_range_half
        cnt += 1
        if  (curr_width_jitter + width_start_index >= 0) and \
            (curr_width_jitter + width_end_index < cube_shape[2]):
            break

    if (curr_depth_jitter + depth_start_index >= 0) and (curr_depth_jitter + depth_end_index < cube_shape[0]):
        split[0][0] = curr_depth_jitter + depth_start_index
        split[0][1] = curr_depth_jitter + depth_end_index
    if (curr_height_jitter + height_start_index >= 0) and (curr_height_jitter + height_end_index < cube_shape[1]):
        split[1][0] = curr_height_jitter + height_start_index
        split[1][1] = curr_height_jitter + height_end_index
    if (curr_width_jitter + width_start_index >= 0) and (curr_width_jitter + width_end_index < cube_shape[2]):
        split[2][0] = curr_width_jitter + width_start_index
        split[2][1] = curr_width_jitter + width_end_index
    log.warning("After the augmentation of jittering, current split = {0}".format(split))
    return split


def augment(image, label, is_flip=True, is_swap=False, is_smooth=False, is_jitter=False):
    r'''
    Make the data augmentation for the cropped image cube and label cube
    Parameters
    ----------
    image : the cropped image cube
    label : the corresponding cropped label cube, i.e. the ground truth
    is_flip : flag for random flipping
    is_swap : flag for random swapping
    is_smooth : flag for Gaussian smoothing on the CT image cube
    is_jitter : flag for intensity jittering on the CT image cube

    Returns
    -------
    augmented training samples
    '''
    if is_swap:
        if image.shape[0] == image.shape[1] and image.shape[0] == image.shape[2]:
            axis_order = np.random.permutation(3)
            image = np.transpose(image, axis_order)
            label = np.transpose(label, axis_order)

    if is_flip:
        flip_id = np.random.randint(2) * 2 - 1
        image = np.ascontiguousarray(image[:, :, ::flip_id])
        label = np.ascontiguousarray(label[:, :, ::flip_id])

    probability = random.random()
    if is_jitter and probability > 0.5:
        ADD_INT = np.random.rand(image.shape[0], image.shape[1], image.shape[2]) * 2 - 1
        ADD_INT = (ADD_INT * 10).astype('float')
        curr_label_ROI = label * ADD_INT / 255.0
        image += curr_label_ROI
        image[image < 0] = 0
        image[image > 1] = 1

    probability = random.random()
    if is_smooth and probability > 0.5:
        sigma = np.random.rand()
        if sigma > 0.5:
            image = gaussian_filter(image, sigma=1.0)

    return image, label


# Classes ==========================================================================================
class ATM22AirwayDataset(Dataset):
    def __init__(self, config, phase='train', split_comber=None, is_randomly_selected=False):
        r'''
        Organize the ATM22 airway dataset

        Parameters
        ----------
        config : configuration from the model
        phase : specify which phase your dataset is working for, only "train", "val" or "test" can be chosen
        split_comber : it is the object to specify how to split-then-combine the dataset
        is_randomly_selected : whether to select the data item randomly
        '''
        super().__init__()

        assert (phase == 'train') or (phase == 'val') or (phase == 'test')
        self.phase = phase
        self.augmentation_type = config['augtype']
        self.split_comber = split_comber
        self.rand_sel = is_randomly_selected
        self.patch_per_case = 5     # patches used per case if random training

        # Specify the path of dataset
        self.dataset_path = config['dataset_path']
        self.dataset = load_pickle(config['dataset_split'])

        print("------------------------------Load all data into memory------------------------------")
        label_list = []
        cube_list = []
        self.caseNum = 0
        all_image_data_in_memory = {}
        all_label_data_in_memory = {}

        if self.phase == "train":
            trainset_data_files = self.dataset['train']
            self.caseNum += len(trainset_data_files)
            print("In {0} phase, total case number: {1}".format(self.phase, self.caseNum))
            self._load_trainset_data_into_memory(trainset_data_files,
                                                 all_image_data_in_memory,
                                                 all_label_data_in_memory,
                                                 label_list,
                                                 cube_list)
        elif self.phase == "val":
            valset_data_files = self.dataset['val']
            self.caseNum += len(valset_data_files)
            print("In {0} phase, total case number: {1}".format(self.phase, self.caseNum))
            self._load_val_or_test_set_data_into_memory(valset_data_files,
                                                        all_image_data_in_memory,
                                                        all_label_data_in_memory,
                                                        cube_list)
        elif self.phase == "test":
            testset_data_files = self.dataset['test']
            self.caseNum += len(testset_data_files)
            print("In {0} phase, total case number: {1}".format(self.phase, self.caseNum))
            self._load_val_or_test_set_data_into_memory(testset_data_files,
                                                        all_image_data_in_memory,
                                                        all_label_data_in_memory,
                                                        cube_list)

        self.all_image_data_memory = all_image_data_in_memory
        self.all_label_data_memory = all_label_data_in_memory
        if self.rand_sel and phase == 'train':
            assert len(cube_list) == self.caseNum
            mean_label_num = np.mean(np.array(label_list))
            print("mean number of label patchs: {0}".format(mean_label_num))
            print("total patches: {0}".format(self.patch_per_case * self.caseNum))

        random.shuffle(cube_list)
        self.cubelist = cube_list

        print("------------------------------Initialization Done------------------------------")
        print("Phase: {0}, total number of cube-list: {1}\n".format(self.phase, len(self.cubelist)))

    # -----------------------------------------------------------------------------------------------
    def _load_trainset_data_into_memory(self,
                                        data_files,
                                        all_image_data_dict,
                                        all_label_data_dict,
                                        label_list,
                                        cube_list):
        for each_case_dict in data_files:
            image_file_path = each_case_dict['image']
            assert os.path.exists(image_file_path) is True
            label_file_path = each_case_dict['label']
            assert  os.path.exists(label_file_path) is True

            images_np, origin, spacing = load_CT_scan_3D_image(image_file_path)
            splits, num_DHW, shape = self.split_comber.split(images_np)
            labels_np, _, _ = load_CT_scan_3D_image(label_file_path)

            case_name = (image_file_path.split('/')[-1]).split("_clean_hu")[0]
            print("Case name: {0} had been splitted into: \t{1} small cubes".format(case_name, len(splits)))

            log.warning("\timage.shape = {0}, image.origin = {1}, image.spacing = {2}mm"
                        .format(images_np.shape, origin, spacing))
            log.warning("\tlabel.shape = {0}".format(labels_np.shape))

            all_image_data_dict[case_name] = [images_np, origin, spacing]
            all_label_data_dict[case_name] = labels_np

            valid_cubes = []
            for n in range(len(splits)):
                # Check whether the sub-cube is suitable
                cur_split = splits[n]
                label_cube = labels_np[cur_split[0][0]:cur_split[0][1],
                                       cur_split[1][0]:cur_split[1][1],
                                       cur_split[2][0]:cur_split[2][1]]

                pixels_count_in_curr_label_cube = np.sum(label_cube)
                label_list.append(pixels_count_in_curr_label_cube)
                # screening out the valid label cube
                if pixels_count_in_curr_label_cube > 0:
                    cur_list = [case_name, cur_split, n, num_DHW, shape, 'Y']
                    log.warning("\tcube #{0}, \tsplit index range: {1}".format(n, cur_split))
                    log.warning("\t           \tin current label cube, it has pixels count: {0}"
                                .format(pixels_count_in_curr_label_cube))
                    valid_cubes.append(cur_list)

            random.shuffle(valid_cubes)
            if self.rand_sel:
                cube_list.append(valid_cubes)
            else:
                cube_list += valid_cubes

    # -----------------------------------------------------------------------------------------------
    def _load_val_or_test_set_data_into_memory(self,
                                               data_files,
                                               all_image_data_dict,
                                               all_label_data_dict,
                                               cube_list):
        for each_case_dict in data_files:
            image_file_path = each_case_dict['image']
            assert os.path.exists(image_file_path) is True
            label_file_path = each_case_dict['label']
            assert os.path.exists(label_file_path) is True

            images_np, origin, spacing = load_CT_scan_3D_image(image_file_path)
            splits, num_DHW, shape = self.split_comber.split(images_np)
            labels_np, _, _ = load_CT_scan_3D_image(label_file_path)

            case_name = (image_file_path.split('/')[-1]).split("_clean_hu")[0]
            print("Case name: {0} had been splitted into: \t{1} small cubes".format(case_name, len(splits)))

            log.warning("\timage.shape = {0}, image.origin = {1}, image.spacing = {2}mm"
                        .format(images_np.shape, origin, spacing))
            log.warning("\tlabel.shape = {0}".format(labels_np.shape))

            all_image_data_dict[case_name] = [images_np, origin, spacing]
            all_label_data_dict[case_name] = labels_np

            for n in range(len(splits)):
                cur_split = splits[n]
                cur_list = [case_name, cur_split, n, num_DHW, shape, 'N']
                log.warning("\tcube #{0}, \tsplit index range: {1}".format(n, cur_split))
                cube_list.append(cur_list)

    #-----------------------------------------------------------------------------------------------
    def __len__(self):
        if self.phase == 'train' and self.rand_sel:
            return self.patch_per_case * self.caseNum
        else:
            return len(self.cubelist)

    # -----------------------------------------------------------------------------------------------
    def __getitem__(self, index):
        r'''
        Parameters
        ----------
        index : index of the batch

        Returns
        -------
        wrapped data tensor and name, shape, origin, in the torch.tensor format
        '''
        start_time = time.time()
        # log.warning("start_time = {0}".format(start_time))
        fraction = int(str(start_time % 1)[2:8])
        # log.warning("fraction = {0}".format(fraction))
        np.random.seed(fraction)

        if self.phase == 'train' and self.rand_sel:
            caseID = index // self.patch_per_case
            caseSplit = self.cubelist[caseID]
            np.random.shuffle(caseSplit)
            curr_list = caseSplit[0]
        else:
            curr_list = self.cubelist[index]

        # The organization of curr_list is:
        # for trainset :       [case_name, cur_split, n, num_DHW, shape, 'Y']
        # for valset/testset : [case_name, cur_split, n, num_DHW, shape, 'N']
        currNameID = curr_list[0]
        currSplit = curr_list[1]
        currSplitID = curr_list[2]
        currNumDHW = curr_list[3]
        currShape = curr_list[4]
        currTransFlag = curr_list[5]

        if  (self.phase == 'train') and (currTransFlag == 'Y') and \
            (self.augmentation_type['split_jitter'] is True):
            # randomly jittering during the training process
            currSplit = augment_jittering_for_imagecube(currSplit, currShape)

        #---------------------------------------------------------------------------------
        # the organization of all_image_data_memory, e.g.
        # all_image_data_memory['ATM_647_000'] = [images_np, origin, spacing]
        curr_case_image_info = self.all_image_data_memory[currNameID]
        curr_images, curr_origin, curr_spacing = curr_case_image_info[0], \
                                                 curr_case_image_info[1], \
                                                 curr_case_image_info[2]
        curr_image_cube = curr_images[currSplit[0][0]:currSplit[0][1],
                                      currSplit[1][0]:currSplit[1][1],
                                      currSplit[2][0]:currSplit[2][1]]
        curr_image_cube = (curr_image_cube.astype(np.float32)) / 255.0

        # ---------------------------------------------------------------------------------
        curr_case_label = self.all_label_data_memory[currNameID]
        curr_case_label = (curr_case_label > 0)
        curr_case_label = curr_case_label.astype('float')
        curr_label_cube = curr_case_label[currSplit[0][0]:currSplit[0][1],
                                          currSplit[1][0]:currSplit[1][1],
                                          currSplit[2][0]:currSplit[2][1]]

        # ---------------------------------------------------------------------------------
        currNameID = [currNameID]
        currSplitID = [currSplitID]
        currNumDHW = np.array(currNumDHW)
        currShape = np.array(currShape)

        # ---------------------------------------------------------------------------------
        # Make the data augmentation here
        if self.phase == 'train' and currTransFlag == 'Y':
            image_cube, label_cube = augment(curr_image_cube, curr_label_cube,
                                             is_flip = self.augmentation_type['flip'],
                                             is_swap = self.augmentation_type['swap'],
                                             is_smooth = self.augmentation_type['smooth'],
                                             is_jitter = self.augmentation_type['jitter'])
            image_cube = image_cube[np.newaxis, ...]
            label_cube = label_cube[np.newaxis, ...]
        else:
            image_cube = curr_image_cube[np.newaxis, ...]
            label_cube = curr_label_cube[np.newaxis, ...]

        log.warning("\nATM22AirwayDataset.__getitem__[{0}]".format(index))
        log.warning("Current case-name = {0}, split-ID = {1}".format(currNameID, currSplitID))
        log.warning("\timage_cube.shape = {0}, label_cube.shape = {1}, "
                    "raw CT 3dimage.spacing = {2}, 3dimage.shape = {3}, num_DHW = {4}\n"
                    .format(image_cube.shape, label_cube.shape, curr_spacing, currShape, currNumDHW))

        return torch.from_numpy(image_cube).float(), \
               torch.from_numpy(label_cube).float(), \
               torch.from_numpy(curr_origin), \
               torch.from_numpy(curr_spacing), \
               currNameID, \
               currSplitID, \
               torch.from_numpy(currNumDHW), \
               torch.from_numpy(currShape)


#===================================================================================================
if __name__ == "__main__":
    from baseline import config

    splitter = SplitterCombiner(crop_cube_size=[80, 192, 304], crop_stride=[64, 96, 152])

    train_dataset = ATM22AirwayDataset(config,
                                       phase='train',
                                       split_comber=splitter,
                                       is_randomly_selected=False)
    print(train_dataset.cubelist)

    image_cube_tensor, \
        label_cube_tensor, \
        origin_tensor, \
        spacing_tensor, \
        case_name, \
        splitID, \
        num_DHW_tensor, \
        shape_tensor = train_dataset[5]

    print(image_cube_tensor.shape)
    print(label_cube_tensor.shape)
    print(origin_tensor)
    print(spacing_tensor)
    print(case_name)
    print(splitID)
    print(num_DHW_tensor)
    print(shape_tensor)