#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File  : utils.py
#
#


import pickle


# Functions ========================================================================================
def save_obj(obj, filename):
    with open(filename + ".pkl", "wb") as fh:
        pickle.dump(obj, fh, pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    with open(filename + ".pkl", "rb") as fh:
        return pickle.load(fh)


# Classes ==========================================================================================



# Main logics ======================================================================================
if __name__ == "__main__":
    pass