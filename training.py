#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File  : training.py
#
#


import sys
import argparse
import logging
import torch
from torch.utils.data import RandomSampler, DataLoader
import datetime
import gc

from models.segment_3d_airway_model import SegAirwayModel
from dataset import dataset_info_path
from dataset import pklfile_train_dataset_info_more_focus_on_high_gen_airway
from dataset import pklfile_train_dataset_info_more_focus_on_low_gen_airway
from dataset import AirwayDataset
from utils import load_obj
from losses import dice_loss


# Objects ==========================================================================================
log = logging.Logger(__name__)
logging.root.setLevel(logging.NOTSET)
log.info("Training the Airway 3D Segment Net model")

train_file_format = ".npy"
windowMin_CT_img_HU = -1000     # the lower limit of CT image HU value
windowMax_CT_img_HU =  600      # the upper limit of CT image HU value
crop_cube_size = (128, 128, 128)

training_freq_swith_between_high_or_low_gen_airways = 10

# Functions ========================================================================================



# Classes ==========================================================================================
class Airway3DSegmentTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
        
        args_parser = self._initArguments()
        self.cli_args = args_parser.parse_args(sys_argv)
        log.info(self.cli_args)
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
    
    #-----------------------------------------------------------------------------------------------
    def main(self):
        log.info("Start training...  {0}"
                 .format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
        
        low_gen_airways_train_dataset, high_gen_airways_train_dataset = self.initTrainDataset()
        # validate_set_data_loader = self.initValidateSetDataLoader()
        
        for epoch in range(1, self.cli_args.epochs):
            train_data_loader = self.prepareTrainDataLoader(epoch, 
                                                            low_gen_airways_train_dataset,
                                                            high_gen_airways_train_dataset)
            # self.doTraining(epoch, train_data_loader)
            
            
    
    #-----------------------------------------------------------------------------------------------
    def _initArguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch-size",
                            help="Batch size to use for training",
                            default=1,
                            type=int)
        parser.add_argument("--num-workers",
                            help="Number of worker processes for background data loader",
                            default=1,
                            type=int)
        parser.add_argument("--epochs",
                            help="Number of epochs to train for",
                            default=100,
                            type=int)
        parser.add_argument("--num-samples-of-each-epoch",
                            help="Specify how many samples will be loaded to train",
                            default=500,
                            type=int)
        return parser
    
    #-----------------------------------------------------------------------------------------------
    def initModel(self):
        segment_model = SegAirwayModel(in_channels=1, out_channels=2)
        
        if self.use_cuda:
            log.info("Using CUDA: {0} devices".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segment_model = torch.nn.DataParallel(segment_model)
                # segment_model = torch.nn.parallel.DistributedDataParallel(segment_model)
            segment_model = segment_model.to(self.device)
            
        return segment_model
    
    #-----------------------------------------------------------------------------------------------
    def initOptimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-5)
    
    #-----------------------------------------------------------------------------------------------
    def doTraining(self, epoch_index, train_data_loader):
        print("Start #{0:04} training at {1}"
              .format(epoch_index, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")),
              end="\r")
        
        for index, batch in enumerate(train_data_loader):
            input_img = batch['image'].float().to(self.device)
            
            groundtruth_foreground = batch['label'].float().to(self.device)
            groundtruth_background = 1 - groundtruth_foreground
            
            foreground_pixel_num = torch.sum(groundtruth_foreground)
            background_pixel_num = torch.sum(groundtruth_background)
            total_pixel_num = foreground_pixel_num + background_pixel_num
            
            fore_pix_perc = foreground_pixel_num / total_pixel_num
            back_pix_perc = background_pixel_num / total_pixel_num
            total_pix_perc_exp = torch.exp(fore_pix_perc) + torch.exp(back_pix_perc)
            
            total_weights = torch.exp(back_pix_perc) / total_pix_perc_exp * torch.eq(groundtruth_foreground, 1).float() + \
                            torch.exp(fore_pix_perc) / total_pix_perc_exp * torch.eq(groundtruth_foreground, 0).float()

            # Move these intermediate tensor objects from GPU to CPU, in order to save GPU memory
            groundtruth_foreground.to('cpu')
            groundtruth_background.to('cpu')
            foreground_pixel_num.to('cpu')
            background_pixel_num.to('cpu')
            total_pixel_num.to('cpu')
            fore_pix_perc.to('cpu')
            back_pix_perc.to('cpu')
            total_pix_perc_exp.to('cpu')
            total_weights.to('cpu')
            
            output_img = self.model(input_img)
            
            loss1 = dice_loss.dice_loss_weights(predict_values=output_img[:, 0, :, :, :],
                                                target_values=groundtruth_background, 
                                                weights=total_weights)
            
            loss2 = dice_loss.dice_loss_power_weights(predict_values=output_img[:, 1, :, :, :],
                                                      target_values=groundtruth_foreground,
                                                      weights=total_weights,
                                                      alpha=2)
            loss = loss1 + loss2
            accuracy = dice_loss.dice_accuracy(output_img[:, 1, :, :, :], groundtruth_foreground)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Up till now, finished one batch training
        
        print("End #{0:04} training at {1}"
              .format(epoch_index, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")),
              end="\n")
        
        del train_data_loader
        gc.collect()
    
    #-----------------------------------------------------------------------------------------------
    def initTrainDataset(self):
        low_gen_airways_dataset_info = load_obj(dataset_info_path +
            pklfile_train_dataset_info_more_focus_on_low_gen_airway[:-4])
        
        low_gen_airways_train_dataset = AirwayDataset(low_gen_airways_dataset_info)
        low_gen_airways_train_dataset.set_para(file_format=train_file_format,
                                               crop_size=crop_cube_size,
                                               windowMin=windowMin_CT_img_HU,
                                               windowMax=windowMax_CT_img_HU,
                                               need_tensor_output=True,
                                               need_transform=True)
        
        high_gen_airways_dataset_info = load_obj(dataset_info_path + 
            pklfile_train_dataset_info_more_focus_on_high_gen_airway[:-4])
        
        high_gen_airways_train_dataset = AirwayDataset(high_gen_airways_dataset_info)
        high_gen_airways_train_dataset.set_para(file_format=train_file_format,
                                                crop_size=crop_cube_size,
                                                windowMin=windowMin_CT_img_HU,
                                                windowMax=windowMax_CT_img_HU,
                                                need_tensor_output=True,
                                                need_transform=True)
        
        return low_gen_airways_train_dataset, high_gen_airways_train_dataset
    
    #-----------------------------------------------------------------------------------------------
    def prepareTrainDataLoader(self, 
                               epoch_index, 
                               low_gen_airways_dataset,
                               high_gen_airways_dataset):
        if ((epoch_index / training_freq_swith_between_high_or_low_gen_airways) // 2 == 0) or \
            (epoch_index >= (self.cli_args.epochs - training_freq_swith_between_high_or_low_gen_airways)):

            print("At epoch #{0}, traing more focus on the low generation airways, "
            # log.info("At epoch #{0}, traing more focus on the low generation airways, "
                     "namely closer to main trachea".format(epoch_index))
            
            low_gen_airways_sampler = RandomSampler(
                data_source=low_gen_airways_dataset,
                num_samples=min(self.cli_args.num_samples_of_each_epoch, 
                                len(low_gen_airways_dataset)),
                replacement=True)
            
            train_data_loader = DataLoader(low_gen_airways_dataset,
                                           batch_size=self.cli_args.batch_size,
                                           sampler=low_gen_airways_sampler,
                                           num_workers=self.cli_args.num_workers,
                                           pin_memory=True,
                                           persistent_workers=(self.cli_args.num_workers > 1))
        else:
            print("At epoch #{0}, training more focus on high generation airways, "
            # log.info("At epoch #{0}, training more focus on high generation airways, "
                     "namely closer to distal brouchus".format(epoch_index))
            
            high_gen_airways_sampler = RandomSampler(
                data_source=high_gen_airways_dataset,
                num_samples=min(self.cli_args.num_samples_of_each_epoch,
                                len(high_gen_airways_dataset)),
                replacement=True)
            
            train_data_loader = DataLoader(high_gen_airways_dataset,
                                           batch_size=self.cli_args.batch_size,
                                           sampler=high_gen_airways_sampler,
                                           num_workers=self.cli_args.num_workers,
                                           pin_memory=True,
                                           persistent_workers=(self.cli_args.num_workers > 1))
        return train_data_loader
    
    #-----------------------------------------------------------------------------------------------
    def initValidateSetDataLoader(self):
        pass


# Main logics ======================================================================================
if __name__ == "__main__":
    app = Airway3DSegmentTrainingApp()
    app.main()