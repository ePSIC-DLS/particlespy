# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:35:23 2018

@author: qzo13262
"""

import segptcls as seg
import numpy as np
from ptcl_class import Particle
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import find_zoneaxis as zone

def ParticleAnalysis(acquisition,process_param,thresh_im=0,particle_list=[],mask=None):
    
    if mask == None:
        labeled = seg.process(acquisition,process_param)
        #labels = np.unique(labeled).tolist() #some labeled number have been removed by "remove_small_holes" function
    else:
        labeled = label(mask)
        
    for region in regionprops(labeled): #'count' start with 1, 0 is background
        p = Particle()
        
        p_im = np.zeros_like(acquisition.data)
        p_im[labeled==region.label] = acquisition.data[labeled==region.label]
        
        #origin = ac_number
        #p.set_origin(origin)
        
        #Set area
        cal_area = region.area*acquisition.axes_manager[0].scale*acquisition.axes_manager[1].scale
        area_units = acquisition.axes_manager[0].units+"^2"
        p.set_area(cal_area,area_units)
        
        #Set zoneaxis
        p.set_zone(zone.find_zoneaxis(p_im))
        
        particle_list.append(p)
        
        if process_param["store_im"]==True:
            ii = np.where(labeled == region.label)
            
            box_x_min = np.min(ii[0])
            box_x_max = np.max(ii[0])
            box_y_max = np.max(ii[1])
            box_y_min = np.min(ii[1])
            pad = 5
            
            p_boxed = acquisition.isig[(box_y_min-pad):(box_y_max+pad),(box_x_min-pad):(box_x_max+pad)]
            p.store_im(p_boxed)
        
    return(particle_list)
    
def param_generator(threshold='otsu',watershed=None,min_size=None,store_im=None):
    params = {}
    params['threshold'] = threshold
    params['watershed'] = watershed
    params['min_size'] = min_size
    params['store_im'] = store_im
    
    return(params)

def plot_area(p_list):
    
    areas = []
    
    for p in p_list:
        areas.append(p.area)
        
    plt.hist(areas)
    plt.xlabel(p_list[0].area_units)
    plt.ylabel("No. of particles")