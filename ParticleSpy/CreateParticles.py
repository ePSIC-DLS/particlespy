# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:35:23 2018

@author: qzo13262
"""

import segptcls as seg
import numpy as np
from ptcl_class import Particle
from skimage.measure import label, regionprops

def ParticleAnalysis(acquisition,process_param,particle_list=[],mask=None):
    
    if mask == None:
        labeled = seg.process(acquisition,process_param)
        #labels = np.unique(labeled).tolist() #some labeled number have been removed by "remove_small_holes" function
    else:
        labeled = label(mask)
        
    for region in regionprops(labeled): #'count' start with 1, 0 is background
        p = Particle()
        #origin = ac_number
        #p.set_origin(origin)
        
        cal_area = region.area*acquisition.axes_manager[0].scale*acquisition.axes_manager[1].scale
        area_units = acquisition.axes_manager[0].units+"^2"
        p.set_area(cal_area,area_units)
        
        particle_list.append(p)
        
    return(particle_list)