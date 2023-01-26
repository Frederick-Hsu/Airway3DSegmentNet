#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : Airway3DSegmentation.py
# Brief     : This python script implements an application to segment the pulmonary 3D airway model.
# Author    : Frederique Hsu (徐赞, frederique.hsu@outlook.com)
# Date      : Fri.  20 Jan. 2023
# Copyright(C)  2023    All rights reserved.
#
#

# system's modules
import os
import sys
from importlib import  import_module

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, kaiming_normal_, constant_, normal_
from torch.backends import cudnn

# user-defined modules
from arguments import InitArguments
from log_switch import log
from utils import Logger
from splitter_combiner import SplitterCombiner
from ATM22_airway_dataset import ATM22AirwayDataset
from train_validate_test_network import train_network

# Functions ========================================================================================


# Classes ==========================================================================================
class Airway3DSegmentation:
    r'''
    Segment the pulmonary 3D airway model, by virtue of the deep learning-based medical
    image (like chest X-ray CT scans) segmentation.
    '''
    def __init__(self, args=None):
        if args is None:
            args = sys.argv[1:]
        arg_parser = InitArguments()
        self.cli_args = arg_parser.parse_args(args)

        # this 'log.warning' is used to print some debug message, and can be turned off if you don't need
        # these verbose message at the full-speed running time. Only need to change log.setLevel() from
        # logging.DEBUG to logging.ERROR
        log.warning(self.cli_args)

        self.init_load_model()
        self.init_optimizer()
        self.prepare_log_dir()

    #-----------------------------------------------------------------------------------------------
    def training(self):
        train_data_loader = self.prepare_train_dataloader()

        self.train_loss_list = []
        self.train_accuracy_list = []
        self.train_sensitivity_list = []
        self.train_dice_list = []

        for epoch in range(self.cli_args.start_epoch, self.cli_args.epochs + 1):
            loss, mean_accuracy, mean_sensitivity, mean_dice, mean_ppv = \
                train_network(epoch,
                              self.airway_seg_model,
                              train_data_loader,
                              self.optimizer,
                              self.cli_args,
                              self.results_dir)

            self.train_loss_list.append(loss)
            self.train_accuracy_list.append(mean_accuracy)
            self.train_sensitivity_list.append(mean_sensitivity)
            self.train_dice_list.append(mean_dice)


    #-----------------------------------------------------------------------------------------------
    def validating(self):
        pass

    #-----------------------------------------------------------------------------------------------
    def testing(self):
        pass

    #-----------------------------------------------------------------------------------------------
    def init_load_model(self):
        module_name = import_module(self.cli_args.model)
        config, net = module_name.get_model(self.cli_args)

        if self.cli_args.resume:
            pretrained_model_checkpoint = torch.load(self.cli_args.resume)

            if self.cli_args.partial_resume:
                # Load the part of weight parameters
                net.load_state_dict(pretrained_model_checkpoint['state_dict'], strict=False)
                log.warning("Partially loading the model's weight parameters Done!")
            else:
                # Fully load the weight parameters
                net.load_state_dict(pretrained_model_checkpoint['state_dict'])
                log.warning("Fully resume Done!")
        else:
            self.init_model_weights_bias(net, init_type='xavier')

        # Try to use the GPU
        if self.cli_args.multi_gpu_parallel and torch.cuda.is_available():
            net = net.cuda()
            cudnn.benchmark = True
            if torch.cuda.device_count() > 1:
                log.warning("Total {0} GPUs were used to compute the neural network in parallel."
                            .format(torch.cuda.device_count()))
                net = torch.nn.DataParallel(net)

        self.config = config
        self.airway_seg_model = net

    #-----------------------------------------------------------------------------------------------
    def init_model_weights_bias(self, model, init_type='xavier'):
        r'''
        Initialize the weights and bias of CNN models, before training.
        Parameters
        ----------
        model : CNN-like neural network
        init_type : the type of initializing weights & bias which user selects from
                    'normal' and 'xavier' options
        '''
        def init_func(net):
            if isinstance(net, nn.Conv3d) or isinstance(net, nn.Linear):
                if init_type == 'normal':
                    normal_(net.weight.data)
                elif init_type == 'xavier':
                    xavier_normal_(net.weight.data)
                else:
                    kaiming_normal_(net.weight.data)

                if net.bias is not None:
                    constant_(net.bias.data, 0)

        print("Initialize network with {0}".format(init_type))
        model.apply(init_func)

    #-----------------------------------------------------------------------------------------------
    def init_optimizer(self):
        self.cli_args.lr_stage = self.config['lr_stage']
        self.cli_args.lr_preset = self.config['lr']

        if not self.cli_args.sgd:
            optimizer = torch.optim.Adam(self.airway_seg_model.parameters(), lr=1e-03)
        else:
            optimizer = torch.optim.SGD(self.airway_seg_model.parameters(), lr=1e-03, momentum=0.9)
        self.optimizer = optimizer

    #-----------------------------------------------------------------------------------------------
    def prepare_train_dataloader(self):
        print("------------------------------Load the dataset for training------------------------------")
        crop_cube_size = self.cli_args.train_cube_size
        crop_stride = self.cli_args.train_stride
        splitter_combiner = SplitterCombiner(crop_cube_size, crop_stride)

        train_dataset = ATM22AirwayDataset(self.config,
                                           phase='train',
                                           split_comber=splitter_combiner,
                                           is_randomly_selected=self.cli_args.randsel)

        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=self.cli_args.batch_size,
                                                        shuffle=True,
                                                        num_workers=self.cli_args.num_workers,
                                                        pin_memory=True)
        return train_data_loader

    def prepare_validate_dataloader(self):
        print("------------------------------Load the dataset for validating------------------------------")
        crop_cube_size = self.cli_args.val_cube_size
        crop_stride = self.cli_args.val_stride
        splitter = SplitterCombiner(crop_cube_size, crop_stride)

        val_dataset = ATM22AirwayDataset(self.config,
                                         phase='val',
                                         split_comber=splitter,
                                         is_randomly_selected=self.cli_args.randsel)

        val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=self.cli_args.batch_size,
                                                      shuffle=False,
                                                      num_workers=self.cli_args.num_workers,
                                                      pin_memory=True)
        return val_data_loader

    # -----------------------------------------------------------------------------------------------
    def prepare_log_dir(self):
        save_dir = self.cli_args.save_dir
        results_dir = os.path.join("results", save_dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        logfile = os.path.join(results_dir, "log.txt")
        sys.stdout = Logger(logfile)

        log_dir = os.path.join(results_dir, 'log')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self.results_dir = results_dir
        self.log_dir = log_dir

# Main logics ======================================================================================
if __name__ == "__main__":
    app = Airway3DSegmentation()
    
    log.warning("Performing the training process...")
    app.training()
    
    log.warning("Performing the validating process...")
    app.validating()
    
    print("Performing the testing process...")
    app.testing()
