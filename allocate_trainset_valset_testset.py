#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File  : allocate_trainset_valset_testset.py
#
#


import pickle
import os
import random
import copy
from utils import save_pickle, load_pickle

# Important variables ==============================================================================
dataset_path = "../LearningTubuleSensitiveCNNs/preprocessed_datasets/"
dataset_images_path = dataset_path + "imagesTr/"
dataset_labels_path = dataset_path + "labelsTr/"

# Functions ========================================================================================


# Classes ==========================================================================================


# Main logics ======================================================================================
# Retrieve the label files
label_files_list = os.listdir(dataset_labels_path)
label_files_list.sort()
# print(label_files_list)

case_names = []
for label_file in label_files_list:
    case_name = label_file.split("_label.nii.gz")[0]
    # print(case_name)
    case_names.append(case_name)

case_names.sort()
print(case_names)

# Retrieve the image files
files_list = os.listdir(dataset_images_path)
# print(files_list)

image_files_list = []
for file in files_list:
    if "_clean_hu.nii.gz" in file:
        image_files_list.append(file)

image_files_list.sort()
# print(image_files_list)

assert len(label_files_list) == len(image_files_list), \
    "Image and Label must have the same file counts"

for index in range(len(case_names)):
    assert image_files_list[index].split("_clean_hu.nii.gz")[0] == case_names[index]
    assert image_files_list[index].split("_clean_hu.nii.gz")[0] == label_files_list[index].split("_label.nii.gz")[0]

# Randomly split the cases, and allocate them into trainset, valset and testset
dataset_dict = dict()
trainset_list = dataset_dict["train"] = []
# print(type(trainset_dict))
valset_list = dataset_dict["val"] = []
testset_list = dataset_dict["test"] = []

# randomly select from the case_names
trainset_portion = int(len(case_names) * 0.7)
valset_portion = int(len(case_names) * 0.1)
testset_portion = int(len(case_names) * 0.2)
print(trainset_portion, valset_portion, testset_portion)

case_names_copy = copy.deepcopy(case_names)

print("\nAllocate for trainset:")
for n in range(trainset_portion):
    random_index = random.randint(0, len(case_names_copy) - 1)
    print(random_index)
    image_file_path = dataset_images_path + image_files_list[random_index]
    label_file_path = dataset_labels_path + label_files_list[random_index]

    trainset_item = dict()
    trainset_item["image"] = image_file_path
    print(image_file_path)
    trainset_item["label"] = label_file_path
    print(label_file_path)
    trainset_list.append(trainset_item)

    del case_names_copy[random_index]
    del image_files_list[random_index]
    del label_files_list[random_index]

print("\nAllocate for valset:")
for n in range(valset_portion):
    random_index = random.randint(0, len(case_names_copy) - 1)
    print(random_index)
    image_file_path = dataset_images_path + image_files_list[random_index]
    label_file_path = dataset_labels_path + label_files_list[random_index]

    valset_item = dict()
    valset_item["image"] = image_file_path
    print(image_file_path)
    valset_item["label"] = label_file_path
    print(label_file_path)
    valset_list.append(valset_item)

    del case_names_copy[random_index]
    del image_files_list[random_index]
    del label_files_list[random_index]

print("\nAllocate for testset")
for n in range(testset_portion):
    random_index = random.randint(0, len(case_names_copy) - 1)
    print(random_index)
    image_file_path = dataset_images_path + image_files_list[random_index]
    label_file_path = dataset_labels_path + label_files_list[random_index]

    testset_item = dict()
    testset_item["image"] = image_file_path
    print(image_file_path)
    testset_item["label"] = label_file_path
    print(label_file_path)
    testset_list.append(testset_item)

    del case_names_copy[random_index]
    del image_files_list[random_index]
    del label_files_list[random_index]

# ------------------
dataset_dict["train"] = trainset_list
dataset_dict["val"] = valset_list
dataset_dict["test"] = testset_list
# print(dataset_dict)
save_pickle(dataset_dict, filename="split_dataset.pkl")

split_dataset = load_pickle(filename="split_dataset.pkl")
print(split_dataset)

# Testing ==========================================================================================
