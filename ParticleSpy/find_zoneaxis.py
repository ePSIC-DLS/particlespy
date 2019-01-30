# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 13:59:52 2018

@author: qzo13262
"""

import numpy as np
import scipy.fftpack as fftim
from skimage import filters, exposure, measure
from scipy.ndimage import label
from math import isclose, degrees, acos
#import matplotlib.pyplot as plt

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
    
    #Subtract central area
    #circle = set_circle(im.shape,int(im.shape[0]/15))
    #fshift_real[circle==True] = 0
    #plt.imshow(fshift_real)
    
    #Rescale FFT - Check rescaling is applicable to all images
    fft_rescaled = np.uint8(exposure.rescale_intensity(fshift_real, in_range=(0,np.max(fshift_real)), out_range='uint8'))
    #plt.imshow(fft_rescaled)
    #print(fft_rescaled[fft_rescaled!=0].mean())
    
    fft_smoothed = filters.gaussian(fft_rescaled,7)
    #plt.imshow(fft_smoothed)
    
    circle = set_circle(im.shape,int(im.shape[0]/10))
    
    #thresh_fft_yen = filters.threshold_yen(fft_smoothed)
    #print(thresh_fft_yen)
    #thresh_fft = fft_smoothed.max()/20
    thresh_fft = fft_smoothed[circle==True].mean()
    #print(thresh_fft)
    fft_thresh = fft_smoothed > thresh_fft
    #plt.imshow(fft_thresh)
    
    #Subtract central area
    
    fft_thresh[circle==True] = 0
    #plt.imshow(fft_thresh)
    
    markers,nr_objects = label(fft_thresh)
    properties = measure.regionprops(markers,fft_smoothed)
    #print(np.where(properties[0].max_intensity))
    #plt.imshow(markers)
    
    if nr_objects < 4:
        #print('Fewer than 6 peaks were found in the FFT.')
        return(None)
        
    #print(nr_objects)
    #plt.imshow(markers)
    #plt.show()
    
    centroids = []
    distances = []
    vectors = []
    max_ind = []
    areas = []
    for x in range(len(properties)):
        centroids.append(properties[x].centroid)
        max_ind.append(np.where(fft_smoothed[markers==x] == properties[x].max_intensity))
        distances.append(np.sqrt((centroids[x][0]-int(im.shape[0]/2))**2+(centroids[x][1]-int(im.shape[0]/2))**2))
        vectors.append((centroids[x][0]-im.shape[0]/2,centroids[x][1]-im.shape[0]/2))
        areas.append((properties[x].area))
        #print(properties[x].max_intensity)
        #print(vectors[x],distances[x],max_ind[x],centroids[x])
        
    #print(max(areas))
    
    min_peak_index = distances.index(min(distances))
    
    peak_groups = {}
    peak_groups[0] = {}
    peak_groups[0][0] = {'Position':properties[min_peak_index].centroid,'Intensity':properties[min_peak_index].max_intensity}
    peak_groups[0][0]['Distance'] = np.sqrt((peak_groups[0][0]['Position'][0]-int(im.shape[0]/2))**2+(peak_groups[0][0]['Position'][1]-int(im.shape[0]/2))**2)
    peak_groups[0][0]['Angle'] = 0
    
    for x in range(0,nr_objects):
        if x != min_peak_index:
            #print("x="+str(x))
            group_flag = False
            for y in range(len(peak_groups)):
                #print("y="+str(y))
                group_distance = np.sqrt((peak_groups[y][0]['Position'][0]-int(im.shape[0]/2))**2+(peak_groups[y][0]['Position'][1]-int(im.shape[0]/2))**2)
                if isclose(distances[x], group_distance, rel_tol=0.1, abs_tol=0.0):
                    peak_groups[y][len(peak_groups[y])] = {'Position':properties[x].centroid,'Distance':distances[x],'Intensity':properties[x].max_intensity}
                    #print(np.dot(vectors[x],vectors[min_peak_index]))
                    if -1<np.dot(vectors[x],vectors[min_peak_index])/(distances[x]*distances[min_peak_index])<1:
                        peak_groups[y][len(peak_groups[y])-1]['Angle'] = degrees(acos(np.dot(vectors[x],vectors[min_peak_index])/(distances[x]*distances[min_peak_index])))
                    elif isclose(np.dot(vectors[x],vectors[min_peak_index])/(distances[x]*distances[min_peak_index]), -1, rel_tol=0.1, abs_tol=0.0):
                        peak_groups[y][len(peak_groups[y])-1]['Angle'] = degrees(acos(-1))
                    elif isclose(np.dot(vectors[x],vectors[min_peak_index])/(distances[x]*distances[min_peak_index]), 1, rel_tol=0.1, abs_tol=0.0):
                        peak_groups[y][len(peak_groups[y])-1]['Angle'] = degrees(acos(1))
                    group_flag = True
                    break
                else:
                    continue
            if group_flag == False:
                num_pg = len(peak_groups)
                peak_groups[num_pg] = {}
                peak_groups[num_pg][0] = {'Position':properties[x].centroid,'Distance':distances[x],'Intensity':properties[x].max_intensity,'Angle':0}
            else:
                continue
    #print(peak_groups)
    
    angles0 = []
    for peak in peak_groups[0]:
        angles0.append(peak_groups[0][peak]['Angle'])
        
    angles0.sort()
    #print(angles0[1],angles0[3],len(peak_groups[0]))
    
    if len(peak_groups)<2:
        peak_groups[1] = {}
        peak_groups[1][0] = {'Position':1,'Distance':1,'Intensity':1,'Angle':0}
    
    dist2 = [peak_groups[0][0]['Distance'],peak_groups[1][0]['Distance']]
    if len(peak_groups[0])==4 and len(peak_groups[1])==4 and isclose(np.max(dist2)/np.min(dist2),1.414,rel_tol=0.1) and max(areas)<2000:
        if isclose(angles0[1],90.0,abs_tol=20.0):
            zone_axis = '001'
        else:
            zone_axis = None
    elif len(peak_groups[0])==6 and max(areas)<2000:
        #print('Test')
        if isclose(angles0[1],60.0,abs_tol=20.0) and isclose(angles0[3],120.0,abs_tol=20.0):
            zone_axis = '111'
        else:
            zone_axis = None
    elif len(peak_groups[0])==4 and len(peak_groups[1])==4 and isclose(np.max(dist2)/np.min(dist2),1.155,rel_tol=0.1) and max(areas)<2000:
        if isclose(angles0[1],90.0,abs_tol=20.0):
            zone_axis = '011'
        else:
            zone_axis = None
    else:
        zone_axis = None
        
    #print(zone_axis)
    return(zone_axis)
    
    
def set_circle(im_dim=(2048,2048),radius=150):
    x, y = np.indices(im_dim)
    center = (im_dim[0]/2, im_dim[1]/2)
    circle = (x - center[0])**2 + (y - center[1])**2 < radius**2
    return(circle)