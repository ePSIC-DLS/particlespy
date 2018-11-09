# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:35:23 2018

@author: qzo13262
"""

import segptcls as seg
import numpy as np
from ptcl_class import Particle, Particle_list
from skimage.measure import label, regionprops, perimeter
import find_zoneaxis as zone
import warnings

def ParticleAnalysis(acquisition,parameters,particle_list=Particle_list(),mask=None):
    """
    Perform segmentation and analysis of images of particles.
    
    Parameters
    ----------
    acquisition: Hyperpsy signal object or list of hyperspy signal objects.
        Hyperpsy signal object containing a nanoparticle image or a list of signal
         objects that contains an image at position 0 and other datasets following.
    process_param: Dictionary of parameters
        The parameters can be input manually in to a dictionary or can be generated
        using param_generator().
    particle_list: List
        List of already analysed particles that the output can be appended
        to.
    mask: Numpy array
        Numpy array of same 2D size as acquisition that contains a mask of presegmented
        particles.
        
    Returns
    -------
    list: List of Particle objects.
    """
    
    #Check if input is list of signal objects or single one
    if isinstance(acquisition,list):
        image = acquisition[0]
        ac_types = []
        for ac in acquisition:
            if ac.metadata.Signal.signal_type == 'EDS_TEM':
                ac_types.append(ac.metadata.Signal.signal_type)
            else:
                warnings.warn("You have input data that does not have a defined signal type and therefore will not be processed."+
                              " You need to define signal_type in the metadata for anything other than the first dataset.")
    else:
        image = acquisition
        ac_types = 'Image only'
    
    if mask == None:
        labeled = seg.process(image,parameters)
        #labels = np.unique(labeled).tolist() #some labeled number have been removed by "remove_small_holes" function
    else:
        labeled = label(mask)
        
    for region in regionprops(labeled): #'count' start with 1, 0 is background
        p = Particle()
        
        p_im = np.zeros_like(image.data)
        p_im[labeled==region.label] = image.data[labeled==region.label]
        
        maskp = np.zeros_like(image.data)
        maskp[labeled==region.label] = 1
        
        #origin = ac_number
        #p.set_origin(origin)
        
        #Set area
        cal_area = region.area*image.axes_manager[0].scale*image.axes_manager[1].scale
        area_units = image.axes_manager[0].units+"^2"
        p.set_area(cal_area,area_units)
        
        #Set shape measures
        peri = perimeter(maskp,neighbourhood=8)
        circularity = 4*3.14159265*p.area/(peri**2)
        p.set_circularity(circularity)
        
        #Set zoneaxis
        p.set_zone(zone.find_zoneaxis(p_im))
        
        #Set mask
        p.set_mask(maskp)
        
        if parameters.store["store_im"]==True:
            store_image(p,image)
            
        if isinstance(ac_types,list):
            for ac in acquisition:
                if ac.metadata.Signal.signal_type == 'EDS_TEM':
                    ac.set_elements(parameters.eds['elements'])
                    ac.add_lines()
                    if parameters.store["store_maps"]==True:
                        store_maps(p,ac)
        
        particle_list.append(p)
        
    return(particle_list)
    
def store_image(particle,image):
    ii = np.where(particle.mask)
            
    box_x_min = np.min(ii[0])
    box_x_max = np.max(ii[0])
    box_y_max = np.max(ii[1])
    box_y_min = np.min(ii[1])
    pad = 5
    
    p_boxed = image.isig[(box_y_min-pad):(box_y_max+pad),(box_x_min-pad):(box_x_max+pad)]
    particle.store_im(p_boxed)
    
def store_maps(particle,ac):
    maps = ac.get_lines_intensity()
    
    for map in maps:
        ii = np.where(particle.mask)
                
        box_x_min = np.min(ii[0])
        box_x_max = np.max(ii[0])
        box_y_max = np.max(ii[1])
        box_y_min = np.min(ii[1])
        pad = 5
        
        p_boxed = map.isig[(box_y_min-pad):(box_y_max+pad),(box_x_min-pad):(box_x_max+pad)]
        particle.store_map(p_boxed,p_boxed.metadata.Sample.elements[0])
    
class parameters(object):
    """A parameters object."""
    
    def generate(self,threshold='otsu',watershed=None,invert=None,min_size=None,store_im=None,rb_kernel=0):
        self.segment = {}
        self.segment['threshold'] = threshold
        self.segment['watershed'] = watershed
        self.segment['invert'] = invert
        self.segment['min_size'] = min_size
        self.segment['rb_kernel'] = rb_kernel
        
        self.store = {}
        self.store['store_im'] = store_im
        
    def generate_eds(self,eds_method=None,elements=None, factors=None):
        self.eds = {}
        self.eds['method'] = eds_method
        self.eds['elements'] = elements
        self.eds['factors'] = factors
        
    
def param_generator(threshold='otsu',watershed=None,invert=None,min_size=None,store_im=None,rb_kernel=0):
    """
    Generate a process parameter dictionary.
        
    Returns
    -------
    dictionary: Parameters contained in dictionary.
    """
    params = {}
    params['threshold'] = threshold
    params['watershed'] = watershed
    params['invert'] = invert
    params['min_size'] = min_size
    params['store_im'] = store_im
    params['rb_kernel'] = rb_kernel
    
    return(params)
