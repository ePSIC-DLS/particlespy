# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:59:52 2018

@author: qzo13262
"""

import numpy as np
import scipy.fftpack as fftim
from skimage import filters, exposure, measure
from scipy.ndimage import label
from math import isclose
import matplotlib.pyplot as plt

def find_zoneaxis(im):
    """
    Determines whether an atomic resolution image is solely of a major zone axis 
    (001, 011, 111) for cubic materials.
    
    Parameters
    ----------
    im: Numpy array
        Atomic resolution image of sample.
        
    Returns
    -------
    string: Zone axis of material in image (eg. '011')
    """
    
    #Get FFT of particle image
    fft = fftim.fft2(im)
    fshift = fftim.fftshift(fft)
    fshift_real = np.real(fshift)
    
    #Rescale FFT - Check rescaling is applicable to all images
    fft_rescaled = np.uint8(exposure.rescale_intensity(fshift_real, in_range=(0,10000), out_range='uint8'))
    
    #Subtract central area
    circle = set_circle(im.shape,150)
    fft_rescaled[circle==True] = 0
    
    fft_smoothed = filters.gaussian(fft_rescaled,21)
    #plt.imshow(fft_smoothed)
    
    thresh_fft_yen = filters.threshold_yen(fft_smoothed)
    fft_thresh_yen = fft_smoothed > thresh_fft_yen
    #plt.imshow(fft_thresh_yen)
    markers,nr_objects = label(fft_thresh_yen)
    properties = measure.regionprops(markers,fft_smoothed)
    
    if nr_objects < 6:
        #print('Fewer than 6 peaks were found in the FFT.')
        return(None)
    
    centroids = []
    for x in range(len(properties)):
        centroids.append(properties[x].centroid)
    
    distances = []
    for x in range(len(properties)):
        distances.append(np.sqrt((centroids[x][0]-1024)**2+(centroids[x][1]-1024)**2))
    
    min_peak_index = distances.index(min(distances))
    
    peak_groups = {}
    peak_groups[0] = {}
    peak_groups[0][0] = {'Position':properties[min_peak_index].centroid,'Intensity':properties[min_peak_index].max_intensity}
    peak_groups[0][0]['Distance'] = np.sqrt((peak_groups[0][0]['Position'][0]-1024)**2+(peak_groups[0][0]['Position'][1]-1024)**2)
    
    for x in range(0,nr_objects):
        if x != min_peak_index:
            #print("x="+str(x))
            group_flag = False
            for y in range(len(peak_groups)):
                #print("y="+str(y))
                group_distance = np.sqrt((peak_groups[y][0]['Position'][0]-1024)**2+(peak_groups[y][0]['Position'][1]-1024)**2)
                if isclose(distances[x], group_distance, rel_tol=0.05, abs_tol=0.0):
                    peak_groups[y][len(peak_groups[y])] = {'Position':properties[x].centroid,'Distance':distances[x],'Intensity':properties[x].max_intensity}
                    group_flag = True
                    break
                else:
                    continue
            if group_flag == False:
                num_pg = len(peak_groups)
                peak_groups[num_pg] = {}
                peak_groups[num_pg][0] = {'Position':properties[x].centroid,'Distance':distances[x],'Intensity':properties[x].max_intensity}
            else:
                continue
        
    dist2 = [peak_groups[0][0]['Distance'],peak_groups[1][0]['Distance']]
    if len(peak_groups[0])==4 and len(peak_groups[1])==4 and isclose(np.max(dist2)/np.min(dist2),1.414,rel_tol=0.05):
        zone_axis = '001'
    elif len(peak_groups[0])==6:
        zone_axis = '111'
    elif len(peak_groups[0])==4 and len(peak_groups[1])==2 and isclose(np.max(dist2)/np.min(dist2),1.155,rel_tol=0.05):
        zone_axis = '011'
    else:
        zone_axis = None
        
    print(zone_axis)
    return(zone_axis)
    
    
def set_circle(im_dim=(2048,2048),radius=150):
    x, y = np.indices(im_dim)
    center = (im_dim[0]/2, im_dim[1]/2)
    circle = (x - center[0])**2 + (y - center[1])**2 < radius**2
    return(circle)