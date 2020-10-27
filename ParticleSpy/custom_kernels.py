# -*- coding: utf-8 -*-
"""
Created on Tues Oct 27 10:13 2020

@author: CBell
"""

import numpy as np
from skimage.transform import rotate
from scipy.ndimage import convolve

def membrane_projection(image):
    kernel = np.zeros([19,19])
    kernel[:,9] = 1
    
    convolved_set = np.zeros([image.shape[0], image.shape[1],30])
    feature_stack = np.zeros([image.shape[0], image.shape[1], 6])

    for r in range(30):
        rot_kernel = rotate(kernel, (r*6))
        convolved_set[:,:,r] = convolve(image,rot_kernel)
        feature_stack[:,:,0] = feature_stack[:,:,0] + convolved_set[:,:,r]

    feature_stack[:,:,0] = np.sum(convolved_set, axis=2)
    feature_stack[:,:,1] = feature_stack[:,:,0]/6
    feature_stack[:,:,2] = np.std(convolved_set, axis=2)
    feature_stack[:,:,3] = np.median(convolved_set, axis=2)
    feature_stack[:,:,4] = np.max(convolved_set, axis=2)
    feature_stack[:,:,5] = np.min(convolved_set, axis=2)

    return feature_stack