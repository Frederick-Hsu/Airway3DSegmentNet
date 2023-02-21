#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# File  : CT_3D_cube_images.py
#
#

import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt

# Functions ========================================================================================
def load_CT_scan_3D_image(niigz_file_name):
    itkimage = sitk.ReadImage(niigz_file_name)
    numpyImages = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImages, numpyOrigin, numpySpacing

# Main logics ======================================================================================
ct_slice_images, origin, spacing = load_CT_scan_3D_image("ATM_054_0000_clean_hu.nii.gz")
print(ct_slice_images.shape)

depth, height, width = ct_slice_images.shape

# X, Y, Z = np.meshgrid(np.arange(width), np.arange(height), np.arange(depth))
# print(X.shape, Y.shape, Z.shape)
# print(X[:, :, 0].shape, Y[:, :, 0].shape, ct_slice_images[256, :, :].shape)
#
# # Create a figure with 3D axes
# fig, axes = plt.subplots(figsize=(20, 20), subplot_kw={"projection": "3d"})
# axes.set_xlabel("width (x)", color='r')
# axes.set_ylabel("height (y)", color='g')
# axes.set_zlabel("depth (z)", color='b')
#
# axes.set(xlim3d=[0, width], ylim3d=[0, height], zlim3d=[0, depth])
#
# for index in np.arange(10, depth, 50):
#     axes.contourf(X[:, :, 0], Y[:, :, 0], ct_slice_images[index, :, :], zdir='z', offset=index, cmap='gray')
#
# plt.show()


X, Y, Z = np.meshgrid(np.arange(height), np.arange(depth), np.arange(width))
print(X.shape, Y.shape, Z.shape)
print(X[0, :, :].shape, Y[0, :, :].shape, ct_slice_images[256, :, :].shape)

# Create a figure with 3D axes
fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
axes.set_xlabel(r"$height$", color='r')
axes.set_ylabel(r"$width$", color='g')
axes.set_zlabel(r"$depth$", color='b')

axes.set(xlim3d=[0, height], ylim3d=[0, width], zlim3d=[0, depth])

for index in [150, 200, 300, 350, 400]:
    axes.contourf(X[0, :, :], Z[0, :, :], ct_slice_images[index, :, :],
                  zdir='z', offset=index, cmap='gray')
#
# axes.contourf(X[0, :, :], Z[0, :, :], ct_slice_images[0, :, :],
#               zdir='z', offset=0, cmap='gray')
# axes.contourf(X[0, :, :], Z[0, :, :], ct_slice_images[depth-1, :, :],
#               zdir='z', offset=depth-1, cmap='gray')
#
#
# axes.contourf(ct_slice_images[:, 0, :], Y[:, 0, :], Z[:, 0, :],
#               zdir='x', offset=0, cmap='gray')
# axes.contourf(ct_slice_images[:, 0, :], Y[:, height-1, :], Z[:, 0, :],
#               zdir='x', offset=height-1, cmap='gray')


plt.show()