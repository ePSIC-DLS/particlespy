# -*- coding: utf-8 -*-
"""
Created on Tues Oct 27 10:13 2020

@author: CBell
"""

import numpy as np
from scipy.ndimage import convolve
from skimage import filters
from skimage.exposure import rescale_intensity
from skimage.transform import rotate


def membrane_projection(image):
    """
    creates a set of features  from a 19by19 kernel with the central column set 
    as ones, rotated through 180 degrees and z projected into one image using 6 
    different methods.

    Parameters
    ----------
    image : greyscale image for feature creation.

    Returns
    -------
    feature_stack: Numy Array
    6 membrane projection features

    """
    kernel = np.zeros([19,19])
    kernel[:,9] = 1
    
    convolved_set = np.zeros([image.shape[0], image.shape[1],30])
    feature_stack = np.zeros([image.shape[0], image.shape[1], 6])

    for r in range(30):
        rot_kernel = rotate(kernel, (r*6))
        convolved_set[:,:,r] = convolve(image,rot_kernel)
        feature_stack[:,:,0] = feature_stack[:,:,0] + convolved_set[:,:,r]

    feature_stack[:,:,1] = feature_stack[:,:,0]/6
    feature_stack[:,:,2] = np.std(convolved_set, axis=2)
    feature_stack[:,:,3] = np.median(convolved_set, axis=2)
    feature_stack[:,:,4] = np.max(convolved_set, axis=2)
    feature_stack[:,:,5] = np.min(convolved_set, axis=2)

    return feature_stack

def laplacian(image):
    """
    creates the laplacian feature

    Parameters
    ----------
    image : greyscale image for feature creation.

    Returns
    -------
    feature_stack: Numpy Array
    6 membrane projection features

    """
    kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    return custom_kernel(kernel=kernel)

def custom_kernel(image, kernel):
    """
    convolves an image with a custom kernel

    Parameters
    ----------
    image : greyscale image for feature creation.

    Returns
    -------
    feature_stack: Numpy Array
    6 membrane projection features

    """
    convolved = convolve(image,kernel)
    return convolved
