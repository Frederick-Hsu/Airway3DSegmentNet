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
import csv
# system's modules
import os
import sys
from importlib import  import_module

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, kaiming_normal_, constant_, normal_
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

# user-defined modules
from arguments import InitArguments
from log_switch import log
from utils import Logger
from splitter_combiner import SplitterCombiner
from ATM22_airway_dataset import ATM22AirwayDataset
from train_validate_test_network import train_network, validate_test_network

# Functions ========================================================================================
def save_model_checkpoint(model, use_multigpu, args, save_dir, checkpoint_name):
    if use_multigpu:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save({'state_dict': state_dict, 'args': args},
               os.path.join(save_dir, checkpoint_name))

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

        self.total_count = 0

    #-----------------------------------------------------------------------------------------------
    def training(self):
        # Initialize the Tensor Board
        self.init_TensorBoard_writers(tensorboard_dir=os.path.join(self.results_dir, 'tensorboard'))

        train_data_loader = self.prepare_train_dataloader()
        val_data_loader = self.prepare_validate_dataloader()
        test_data_loader = self.prepare_test_dataloader()

        self.train_loss_list = []
        self.train_accuracy_list = []
        self.train_sensitivity_list = []
        self.train_dice_list = []
        self.train_ppv_list = []

        self.val_loss_list = []
        self.val_accuracy_list = []
        self.val_sensitivity_list = []
        self.val_dice_list = []
        self.val_ppv_list = []

        self.test_loss_list = []
        self.test_accuracy_list = []
        self.test_sensitivity_list = []
        self.test_dice_list = []
        self.test_ppv_list = []

        self.train_epoch_list = []
        self.val_epoch_list = []
        self.test_epoch_list = []

        for epoch in range(self.cli_args.start_epoch, self.cli_args.epochs + 1):
            train_mean_loss, train_mean_accuracy, train_mean_sensitivity, train_mean_dice, train_mean_ppv = \
                train_network(epoch,
                              model=self.airway_seg_model,
                              data_loader=train_data_loader,
                              optimizer=self.optimizer,
                              args=self.cli_args,
                              tensorboard_writer=self.tensorboard_writer,
                              total_count=self.total_count)

            self.tensorboard_writer.add_scalar(tag='train_mean_loss', scalar_value=train_mean_loss, global_step=epoch)
            self.tensorboard_writer.add_scalar(tag='train_mean_accuracy', scalar_value=train_mean_accuracy, global_step=epoch)
            self.tensorboard_writer.add_scalar(tag='train_mean_sensitivity', scalar_value=train_mean_sensitivity, global_step=epoch)
            self.tensorboard_writer.add_scalar(tag='train_mean_dice', scalar_value=train_mean_dice, global_step=epoch)
            self.tensorboard_writer.add_scalar(tag='train_mean_positive_probability', scalar_value=train_mean_ppv, global_step=epoch)
            self.tensorboard_writer.flush()

            self.train_loss_list.append(train_mean_loss)
            self.train_accuracy_list.append(train_mean_accuracy)
            self.train_sensitivity_list.append(train_mean_sensitivity)
            self.train_dice_list.append(train_mean_dice)
            self.train_ppv_list.append(train_mean_ppv)
            self.train_epoch_list.append(epoch)

            save_model_checkpoint(model=self.airway_seg_model,
                                  use_multigpu=self.cli_args.multi_gpu_parallel,
                                  args=self.cli_args,
                                  save_dir=self.results_dir,
                                  checkpoint_name="model_latest.ckpt")

            if epoch % self.cli_args.save_freq == 0:
                save_model_checkpoint(model=self.airway_seg_model,
                                      use_multigpu=self.cli_args.multi_gpu_parallel,
                                      args=self.cli_args,
                                      save_dir=self.results_dir,
                                      checkpoint_name="model_{0:03}.ckpt".format(epoch))

            #---------------------------------------------------------------------------------------
            if (epoch == self.cli_args.start_epoch) or (epoch % self.cli_args.val_freq == 0):
                val_dir = os.path.join(self.results_dir, 'val{0:03}'.format(epoch))
                if not os.path.exists(val_dir):
                    os.mkdir(val_dir)

                val_mean_loss, val_mean_accuracy, val_mean_sensitivity, val_mean_dice, val_mean_ppv = \
                    validate_test_network(epoch,
                                          phase='val',
                                          model=self.airway_seg_model,
                                          data_loader=val_data_loader,
                                          args=self.cli_args,
                                          save_dir=val_dir,
                                          tensorboard_writer=self.tensorboard_writer,
                                          total_count=self.total_count)

                self.tensorboard_writer.add_scalar(tag='val_mean_loss', scalar_value=val_mean_loss, global_step=epoch)
                self.tensorboard_writer.add_scalar(tag='val_mean_accuracy', scalar_value=val_mean_accuracy, global_step=epoch)
                self.tensorboard_writer.add_scalar(tag='val_mean_sensitivity', scalar_value=val_mean_sensitivity, global_step=epoch)
                self.tensorboard_writer.add_scalar(tag='val_mean_dice', scalar_value=val_mean_dice, global_step=epoch)
                self.tensorboard_writer.add_scalar(tag='val_mean_positive_probability', scalar_value=val_mean_ppv, global_step=epoch)
                self.tensorboard_writer.flush()

                self.val_loss_list.append(val_mean_loss)
                self.val_accuracy_list.append(val_mean_accuracy)
                self.val_sensitivity_list.append(val_mean_sensitivity)
                self.val_dice_list.append(val_mean_dice)
                self.val_ppv_list.append(val_mean_ppv)
                self.val_epoch_list.append(epoch)

            #---------------------------------------------------------------------------------------
            if epoch % self.cli_args.test_freq == 0:
                test_dir = os.path.join(self.results_dir, "test{0:03}".format(epoch))
                if not os.path.exists(test_dir):
                    os.mkdir(test_dir)

                # both val and test phase call the same function "validate_test_network",
                # only 'phase' and 'data_loader' arguments are different
                test_mean_loss, test_mean_accuracy, test_mean_sensitivity, test_mean_dice, test_mean_ppv = \
                    validate_test_network(epoch,
                                          phase='test',
                                          model=self.airway_seg_model,
                                          data_loader=test_data_loader,
                                          args=self.cli_args,
                                          save_dir=test_dir,
                                          tensorboard_writer=self.tensorboard_writer,
                                          total_count=self.total_count)

                self.tensorboard_writer.add_scalar(tag='test_mean_loss', scalar_value=test_mean_loss, global_step=epoch)
                self.tensorboard_writer.add_scalar(tag='test_mean_accuracy', scalar_value=test_mean_accuracy, global_step=epoch)
                self.tensorboard_writer.add_scalar(tag='test_mean_sensitivity', scalar_value=test_mean_sensitivity, global_step=epoch)
                self.tensorboard_writer.add_scalar(tag='test_mean_dice', scalar_value=test_mean_dice, global_step=epoch)
                self.tensorboard_writer.add_scalar(tag='test_mean_positive_probability', scalar_value=test_mean_ppv, global_step=epoch)
                self.tensorboard_writer.flush()

                self.test_loss_list.append(test_mean_loss)
                self.test_accuracy_list.append(test_mean_accuracy)
                self.test_sensitivity_list.append(test_mean_sensitivity)
                self.test_dice_list.append(test_mean_dice)
                self.test_ppv_list.append(test_mean_ppv)
                self.test_epoch_list.append(epoch)

        self._save_metrics()
        self.tensorboard_writer.close()
        print("Training DONE!")

    def _save_metrics(self):
        train_metrics_summary = np.array([self.train_epoch_list,
                                          self.train_loss_list,
                                          self.train_accuracy_list,
                                          self.train_sensitivity_list,
                                          self.train_dice_list,
                                          self.train_ppv_list])
        np.save(os.path.join(self.log_dir, "train_metrics_log.npy"), train_metrics_summary)

        val_metrics_summary = np.array([self.val_epoch_list,
                                        self.val_loss_list,
                                        self.val_accuracy_list,
                                        self.val_sensitivity_list,
                                        self.val_dice_list,
                                        self.val_ppv_list])
        np.save(os.path.join(self.log_dir, "val_metrics_log.npy"), val_metrics_summary)

        test_metrics_summary = np.array([self.test_epoch_list,
                                         self.test_loss_list,
                                         self.test_accuracy_list,
                                         self.test_sensitivity_list,
                                         self.test_dice_list,
                                         self.test_ppv_list])
        np.save(os.path.join(self.log_dir, "test_metrics_log.npy"), test_metrics_summary)

        # -------------------------------------------------------------------------------------------
        logName = os.path.join(self.log_dir, "total_metrics_log.csv")
        with open(logName, 'w') as csv_fh:
            writer = csv.writer(csv_fh)

            title_row = ['phase', 'epoch', 'loss', 'accuracy', 'sensitivity', 'dice', 'positive probability']
            writer.writerow(title_row)

            for index in range(len(self.train_epoch_list)):
                row = ['train',
                       self.train_epoch_list[index],
                       self.train_loss_list[index],
                       self.train_accuracy_list[index],
                       self.train_sensitivity_list[index],
                       self.train_dice_list[index],
                       self.train_ppv_list[index]]
                writer.writerow(row)

            for index in range(len(self.val_epoch_list)):
                rwo = ['validate',
                       self.val_epoch_list[index],
                       self.val_loss_list[index],
                       self.val_accuracy_list[index],
                       self.val_sensitivity_list[index],
                       self.val_dice_list[index],
                       self.val_ppv_list[index]]
                writer.writerow(row)

            for index in range(len(self.test_epoch_list)):
                row = ['test',
                       self.test_epoch_list[index],
                       self.test_loss_list[index],
                       self.test_accuracy_list[index],
                       self.test_sensitivity_list[index],
                       self.test_dice_list[index],
                       self.test_ppv_list[index]]
                writer.writerow(row)

    #-----------------------------------------------------------------------------------------------
    def validating(self):
        val_data_loader = self.prepare_validate_dataloader()

        validate_dir = os.path.join(self.results_dir, 'validate')
        if not os.path.exists(validate_dir):
            os.mkdir(validate_dir)

        self.init_TensorBoard_writers(os.path.join(validate_dir, 'tensorboard'))

        epoch = 1   # Only need to carry out 1 epoch of "validate_network()"
        val_mean_loss, val_mean_accuracy, val_mean_sensitivity, val_mean_dice, val_mean_ppv = \
            validate_test_network(epoch,
                                  phase='val',
                                  model=self.airway_seg_model,
                                  data_loader=val_data_loader,
                                  args=self.cli_args,
                                  save_dir=validate_dir,
                                  tensorboard_writer=self.tensorboard_writer,
                                  total_count=self.total_count)
        self.tensorboard_writer.close()
        print("Validating DONE!")

    #-----------------------------------------------------------------------------------------------
    def testing(self):
        test_data_loader = self.prepare_test_dataloader()

        test_dir = os.path.join(self.results_dir, 'test')
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

        self.init_TensorBoard_writers(os.path.join(test_dir, 'tensorboard'))

        epoch = 1
        test_mean_loss, test_mean_accuracy, test_mean_sensitivity, test_mean_dice, test_mean_ppv = \
            validate_test_network(epoch,
                                  phase='test',
                                  model=self.airway_seg_model,
                                  data_loader=test_data_loader,
                                  args=self.cli_args,
                                  save_dir=test_dir,
                                  tensorboard_writer=self.tensorboard_writer,
                                  total_count=self.total_count)
        self.tensorboard_writer.close()
        print("Testing DONE!")

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
        if torch.cuda.is_available():
            net = net.cuda()
            cudnn.benchmark = True
        if (self.cli_args.multi_gpu_parallel) and (torch.cuda.device_count() > 1):
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

    def init_TensorBoard_writers(self, tensorboard_dir):
        self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
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
                                         is_randomly_selected=False)

        val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=self.cli_args.batch_size,
                                                      shuffle=False,
                                                      num_workers=self.cli_args.num_workers,
                                                      pin_memory=True)
        return val_data_loader

    def prepare_test_dataloader(self):
        print("------------------------------Load the dataset for testing------------------------------")
        # Use the same crop_cube_size and crop_stride with validate dataset
        crop_cube_size = self.cli_args.val_cube_size
        crop_stride = self.cli_args.val_stride
        splitter = SplitterCombiner(crop_cube_size, crop_stride)

        test_dataset = ATM22AirwayDataset(self.config,
                                          phase='test',
                                          split_comber=splitter,
                                          is_randomly_selected=False)

        test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=self.cli_args.batch_size,
                                                       shuffle=False,
                                                       num_workers=self.cli_args.num_workers,
                                                       pin_memory=True)
        return test_data_loader

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

    if app.cli_args.enable_training:
        print("Performing the training process...")
        app.training()

    if app.cli_args.enable_validating:
        print("Performing the validating process...")
        app.validating()

    if app.cli_args.enable_testing:
        print("Performing the testing process...")
        app.testing()
