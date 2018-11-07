# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:06:08 2018

@author: qzo13262
"""

import numpy as np
import scipy.ndimage as ndi

from skimage.filters import threshold_otsu, threshold_mean, threshold_minimum
from skimage.filters import threshold_yen, threshold_isodata, threshold_li
from skimage.filters import threshold_local

from skimage.measure import label
from skimage.morphology import remove_small_objects, watershed, square, white_tophat
from skimage.segmentation import clear_border
from skimage.feature import peak_local_max
from skimage.util import invert

def process(im, param):
    """
    Perform segmentation of an image of particles.
    
    Parameters
    ----------
    im: Hyperpsy signal object
        Hyperpsy signal object containing a nanoparticle image.
    process_param: Dictionary of parameters
        The parameters for segmentation.
        
    Returns
    -------
    numpy array: Labels corresponding to particles in the image.
    """
    
    if isinstance(im,(list,)):
        data = im[0].data
    else:
        data = im.data
    
    data = rolling_ball(data,param.segment["rb_kernel"])
    
    if param.segment["invert"]!=None:
        data = invert(data)
        
    if param.segment["threshold"]!=None:
        labels = threshold(data, param.segment)
        
    labels = clear_border(labels)
    
    if param.segment["watershed"]!=None:
        labels = p_watershed(labels)
        
    if param.segment["min_size"]!=None:
        remove_small_objects(labels,param.segment["min_size"],in_place=True)
        
    return(labels)
    
def threshold(data, process_param):
    if process_param["threshold"] == "otsu":
        thresh = threshold_otsu(data)
    if process_param["threshold"] == "mean":
        thresh = threshold_mean(data)
    if process_param["threshold"] == "minimum":
        thresh = threshold_minimum(data)
    if process_param["threshold"] == "yen":
        thresh = threshold_yen(data)
    if process_param["threshold"] == "isodata":
        thresh = threshold_isodata(data)
    if process_param["threshold"] == "li":
        thresh = threshold_li(data)
    if process_param["threshold"] == "local":
        thresh = threshold_local(data,21)
            
    mask = data > thresh
    
    labels = label(mask)
    
    return(labels)
    
def p_watershed(thresh_image):
    distance = ndi.distance_transform_edt(thresh_image)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=thresh_image)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=thresh_image)
    return(labels)
    
def rolling_ball(img,kernelsize=0):
    if kernelsize == 0:
        new_img = img
    else:
        new_img = white_tophat(img,selem=square(kernelsize))
    return (new_img)
