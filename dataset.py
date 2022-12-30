#!/user/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File  : dataset.py
#
#


import os
from torch.utils.data import Dataset
import numpy as np
from skimage import io
import h5py
import torch
import torchio as tio

# Objects ==========================================================================================

dataset_info_path = "./dataset_info/"   # Please use the relative path
if not os.path.exists(dataset_info_path):
    os.mkdir(dataset_info_path)
    
dataset_info_pklfiles = os.listdir(dataset_info_path)
for pklfile in dataset_info_pklfiles:
    if "more_low_gen" in pklfile:
        # print(pklfile)
        pklfile_train_dataset_info_more_focus_on_low_gen_airway = pklfile
    elif "more_high_gen" in pklfile:
        # print(pklfile)
        pklfile_train_dataset_info_more_focus_on_high_gen_airway = pklfile

print(pklfile_train_dataset_info_more_focus_on_high_gen_airway)
print(pklfile_train_dataset_info_more_focus_on_low_gen_airway)


# Functions ========================================================================================



# Classes ==========================================================================================
class Random3DCrop_np:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)), \
               "Attention: Random 3D crop output size should be an int or a tuple (length = 3)"

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert (len(output_size) == 3), \
                   "Attention: Random 3D crop output size: a tuple (length = 3)"
            self.output_size = output_size
    
    #-----------------------------------------------------------------------------------------------
    def random_crop_start_point(self, input_size):
        assert (len(input_size) == 3), \
               "Attention: Random 3D crop output size: a tuple (length = 3)"
        depth,     height,     width = input_size
        depth_new, height_new, width_new = self.output_size
        
        depth_new  = min(depth, depth_new)
        height_new = min(height, height_new)
        width_new  = min(width, width_new)
        
        assert (depth > depth_new and height > height_new and width > width_new), \
               "Attention: input_size should >= crop_size.\n" \
               "now input_size is " + str((depth, height_new, width_new)) + ", while output_size is " + \
               str((depth_new, height_new, width_new))
        
        depth_start  = np.random.randint(0, depth  - depth_new)
        height_start = np.random.randint(0, height - height_new)
        width_start  = np.random.randint(0, width  - width_new)
        
        return depth_start, height_start, width_start
    
    #-----------------------------------------------------------------------------------------------
    def __call__(self, img_3d, start_points=None):
        img_3d = np.array(img_3d)
        
        depth,     height,     width     = img_3d.shape
        depth_new, height_new, width_new = self.output_size
        
        if start_points == None:
            start_points = self.random_crop_start_point(img_3d.shape)
        
        depth_start, height_start, width_start = start_points
        depth_end = min(depth_start + depth_new, depth)
        height_end = min(height_start + height_new, height)
        width_end = min(width_start + width_new, width)
        
        crop_cube = img_3d[depth_start:depth_end, height_start:height_end, width_start:width_end]
        return crop_cube

#===================================================================================================
class AirwayDataset(Dataset):
    def __init__(self, data_dict, num_of_samples=None):
        super().__init__()
        
        self.data_dict = data_dict
        if num_of_samples is not None:
            num_of_samples = min(len(data_dict), num_of_samples)
            
            chosen_names = np.random.choice(np.array(list(data_dict)), 
                                            size=num_of_samples,
                                            replace=False)
        else:
            chosen_names = np.array(list(data_dict))
        
        self.name_list = chosen_names
        self.para = {}
        self.set_para()        
    
    #-----------------------------------------------------------------------------------------------
    def __len__(self):
        return len(self.name_list)
    
    #-----------------------------------------------------------------------------------------------
    def __getitem__(self, index):
        return self.get(index,
                        file_format=self.para['file_format'],
                        crop_size=self.para['crop_size'],
                        windowMin=self.para['windowMin'],
                        windowMax=self.para['windowMax'],
                        need_tensor_output=self.para['need_tensor_output'],
                        need_transform=self.para['need_transform'])
    
    #-----------------------------------------------------------------------------------------------
    def set_para(self, 
                 file_format=".npy", 
                 crop_size=32, 
                 windowMin=-1000, 
                 windowMax=150,
                 need_tensor_output=True,
                 need_transform=True):
        self.para['file_format'] = file_format
        self.para['crop_size'] = crop_size
        self.para['windowMin'] = windowMin
        self.para['windowMax'] = windowMax
        self.para['need_tensor_output'] = need_tensor_output
        self.para['need_transform'] = need_transform
    
    #-----------------------------------------------------------------------------------------------
    def get(self,
            index,
            file_format=".npy",
            crop_size=32,
            windowMin=-1000,
            windowMax=150,
            need_tensor_output=True,
            need_transform=True):
        random3dcrop = Random3DCrop_np(crop_size)
        normalization = Normalization_np(windowMin, windowMax)
        
        name = self.name_list[index]
        
        if file_format == ".npy":
            raw_img = np.load(self.data_dict[name]['image'])
            label_img = np.load(self.data_dict[name]['label'])
        elif file_format == ".nii.gz":
            raw_img = io.imread(self.data_dict[name]['image'], plugin='simpleitk')
            label_img = io.imread(self.data_dict[name]['label'], plugin='simpleitk')
        elif file_format == ".h5":
            hf = h5py.File(self.data_dict[name]['path'], 'r+')
            raw_img = np.array(hf['image'])
            label_img = np.array(hf['label'])
            hf.close()
        
        assert raw_img.shape == label_img.shape
        
        start_points = random3dcrop.random_crop_start_point(raw_img.shape)
        raw_img_crop = random3dcrop(np.array(raw_img, float), start_points)
        label_img_crop = random3dcrop(np.array(label_img, float), start_points)
        raw_img_crop = normalization(raw_img_crop)
        
        raw_img_crop = np.expand_dims(raw_img_crop, axis=0)
        label_img_crop = np.expand_dims(label_img_crop, axis=0)
        
        output = {'image': raw_img_crop, 'label': label_img_crop}
        if need_tensor_output:
            output = self.to_tensor(output)
            if need_transform:
                output = self.transform_to_tensor(output, prob=0.5)
        
        return output
    
    #-----------------------------------------------------------------------------------------------
    def to_tensor(self, images):
        for key in images.keys():
            images[key] = torch.from_numpy(images[key]).float()
        return images
    
    #-----------------------------------------------------------------------------------------------
    def transform_to_tensor(self, image_tensors, prob=0.5):
        dict_img_tio = dict()
        
        for key in image_tensors.keys():
            dict_img_tio[key] = tio.ScalarImage(tensor=image_tensors[key])
        
        subject_all_imgs = tio.Subject(dict_img_tio)
        transform_shape = tio.Compose([tio.RandomFlip(axes=int(np.random.randint(3, size=1)[0]), p=prob),
                                       tio.RandomAffine(p=prob)])
        transformed_subject_all_imgs = transform_shape(subject_all_imgs)
        
        transform_val = tio.Compose([tio.RandomBlur(p=prob),
                                     tio.RandomNoise(p=prob),
                                     tio.RandomMotion(p=prob),
                                     tio.RandomBiasField(p=prob),
                                     tio.RandomSpike(p=prob),
                                     tio.RandomGhosting(p=prob)])
        transformed_subject_all_imgs['image'] = transform_val(transformed_subject_all_imgs['image'])
        
        for key in subject_all_imgs.keys():
            image_tensors[key] = transformed_subject_all_imgs[key].data
        
        return image_tensors


#===================================================================================================
class Normalization_np:
    def __init__(self, windowMin, windowMax):
        self.name = "ManualNormalization"
        
        assert isinstance(windowMin, (int, float))
        assert isinstance(windowMax, (int, float))
        
        self.windowMin = windowMin
        self.windowMax = windowMax
    
    #-----------------------------------------------------------------------------------------------
    def __call__(self, img_3d):
        img_3d_norm = np.clip(img_3d, self.windowMin, self.windowMax)
        img_3d_norm -= np.min(img_3d_norm)
        max_99_val = np.percentile(img_3d_norm, 99)
        
        if max_99_val > 0:
            img_3d_norm = img_3d_norm / max_99_val * 255
        return img_3d_norm

# Main logics ======================================================================================
if __name__ == "__main__":
    from utils import load_obj
    
    dataset_info_dict = load_obj(dataset_info_path + pklfile_train_dataset_info_more_focus_on_high_gen_airway[:-4])
    
    dataset = AirwayDataset(dataset_info_dict)
    output = dataset.get(10)
    
    for name in output.keys():
        print(name, output[name].shape)
    
    dataset.set_para(file_format=".npy",
                     crop_size=64,
                     windowMin=-1000,
                     windowMax=150,
                     need_tensor_output=True,
                     need_transform=True)
    
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=4,
                                                 shuffle=True,
                                                 num_workers=2,
                                                 pin_memory=False,
                                                 persistent_workers=False)
    next_batch = next(iter(dataset_loader))
    for name in next_batch.keys():
        print(name, next_batch[name].shape)