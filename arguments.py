#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File      : arguments.py
# Brief     : collect and process the CLI arguments and options
#
#

import argparse


# Functions ========================================================================================

def InitArguments():
    parser = argparse.ArgumentParser("Here to collect all CLI arguments and options for airway 3D segmentation task.")
    parser.add_argument("--model",
                        metavar="MODEL",
                        help="select your model",
                        default="baseline")
    parser.add_argument("--num-workers",
                        metavar="N",
                        help="number of worker process (default: 8)",
                        type=int,
                        default=8)
    parser.add_argument("--epochs",
                        help="number of total epochs to run",
                        type=int,
                        default=None)
    parser.add_argument("--start-epoch",
                        help="manual epoch number (useful on restart)",
                        type=int,
                        default=1)
    parser.add_argument("--batch-size",
                        help="batch size of data loading",
                        type=int,
                        default=1)
    parser.add_argument("--learning-rate", "--lr",
                        help="initial learning rate",
                        metavar="LR",
                        type=float,
                        default=None)

    parser.add_argument("--save-freq",
                        help="specify the frequency of saving model's checkpoint, "
                             "i.e. how many epochs to save one checkpoint of model state during the training process?",
                        type=int,
                        default=5)
    parser.add_argument("--val-freq",
                        help="validation frequency, i.e. how many epochs to make one validation action?",
                        type=int,
                        default=10)
    parser.add_argument("--test-freq",
                        help="testing frequency, i.e. how many epochs to make a testing action?",
                        type=int,
                        default=10)

    parser.add_argument("--resume",
                        metavar="PATH",
                        help="where to load the latest checkpoint of model's state? (default: none)",
                        type=str,
                        default="")
    parser.add_argument("--partial-resume",
                        help="resume the parameters part of the model",
                        type=int,
                        default=0)
    parser.add_argument("--save-dir",
                        metavar="SAVE_DIRECTORY",
                        help="specify which directory to save the checkpoint of model state (default: none)",
                        type=str,
                        default="")

    parser.add_argument("--enable-test",
                        help="1 for enabling testing and evaluation, 0 for disabling",
                        type=int,
                        default=0)
    parser.add_argument("--enable-debug-mode",
                        metavar="TRUE_OR_FALSE",
                        help="enable or disable the debug mode",
                        type=bool,
                        default=False)
    parser.add_argument("--enable-val-debug",
                        help="whether to enable the debug mode for validation or not",
                        type=bool,
                        default=False)
    parser.add_argument("--enable-dataloader-debug",
                        help="whether to enable the debug mode while loading data",
                        type=bool,
                        default=False)

    parser.add_argument("--randsel",
                        metavar="1 or 0",
                        help="randomly select samples for training",
                        type=int,
                        default=0)
    parser.add_argument("--save-feature",
                        help="save the SAD features or not",
                        type=bool,
                        default=False)
    parser.add_argument("--encoder-path-ad",
                        help="whether to enable the Attention Distillation function or not "
                             "in encoder down-sampling path?",
                        type=bool,
                        default=False)
    parser.add_argument("--decoder-path-ad",
                        help="whether to enable the Attention Distillation function or not "
                             "in the decoder up-sampling path?",
                        type=bool,
                        default=False)
    parser.add_argument("--deep-supervision",
                        metavar="DEEP_SUPERVISION",
                        help="whether to use the deep supervision as auxiliary tasks or not",
                        type=bool,
                        default=False)
    parser.add_argument("--sgd",
                        metavar="SGD_Optimizer",
                        help="whether to use the SGD optimizer",
                        type=int,
                        default=0)
    parser.add_argument("--multi-gpu-parallel",
                        help="whether to use multiple GPUs to train in parallel or not?",
                        type=bool,
                        default=False)

    parser.add_argument("--train-cube-size",
                        help="specify the size of cropped cube for training",
                        type=int,
                        default=[128, 128, 128],
                        nargs="*")
    parser.add_argument("--val-cube-size",
                        help="specify the size of cropped cube for validation",
                        type=int,
                        default=None,
                        nargs="*")
    parser.add_argument("--train-stride",
                        help="specify the stride of sliding window when cropping the cube for training",
                        metavar="stride",
                        type=int,
                        default=[64, 64, 64],
                        nargs="*")
    parser.add_argument("--val-stride",
                        help="specify the stride of sliding window when cropping the cube for validation",
                        metavar="stride",
                        type=int,
                        default=[48, 80, 80],
                        nargs="*")
    
    return parser



#===================================================================================================
if __name__ == "__main__":
    parser = InitArguments()
    args = parser.parse_args()
    print(parser)