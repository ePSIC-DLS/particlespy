# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:23:59 2018

@author: qzo13262
"""

import h5py
from ParticleSpy.ptcl_class import Particle, Particle_list
import hyperspy as hs
import numpy as np

def save_plist(p_list,filename):
    f = h5py.File(filename,'w')
    
    for i, particle in enumerate(p_list.list):
        p_group = f.create_group("Particle "+str(i))
        
	p_group.attrs["Area"] = particle.properties['area']['value']
        p_group.attrs["Area units"] = particle.properties['area']['units']
        p_group.attrs["Equivalent circular diameter"] = particle.properties['equivalent circular diameter']['value']
        p_group.attrs["Equivalent circular diameter units"] = particle.properties['equivalent circular diameter']['units']
        p_group.attrs["X"] = particle.properties['x']['value']
        p_group.attrs["X units"] = particle.properties['x']['units']
        p_group.attrs["Y"] = particle.properties['y']['value']
        p_group.attrs["Y units"] = particle.properties['y']['units']
        p_group.attrs["Major axis length"] = particle.properties['major axis length']['value']
        p_group.attrs["Major axis length units"] = particle.properties['major axis length']['units']
        p_group.attrs["Minor axis length"] = particle.properties['minor axis length']['value']
        p_group.attrs["Minor axis length units"] = particle.properties['minor axis length']['units']
        p_group.attrs["Circularity"] = particle.properties['circularity']['value']
        p_group.attrs["Eccentricity"] = particle.properties['eccentricity']['value']
        p_group.attrs["Solidity"] = particle.properties['solidity']['value']
        p_group.attrs["Intensity"] = particle.properties['intensity']['value']
        p_group.attrs["Intensity_max"] = particle.properties['intensity_max']['value']
        p_group.attrs["Intensity_std"] = particle.properties['intensity_std']['value']


        #p_group.attrs["Zone"] = particle.zone
        
        p_group.create_dataset("Mask",data=particle.mask)
        
        if hasattr(particle, 'image'):
            p_group.create_dataset("Image",data=particle.image.data)
        
    f.close()
        
def load_plist(filename):
    f = h5py.File(filename,'r')
    p_list = Particle_list()
    
    for p_name in list(f.keys()):
        if p_name[:8] == 'Particle':
            p_group = f[p_name]
            particle = Particle()
            
            particle.set_area(p_group.attrs['Area'],p_group.attrs['Area units'])
            particle.set_circularity(p_group.attrs["Circularity"])
            particle.set_circdiam(p_group.attrs["Equivalent circular diameter"],p_group.attrs["Equivalent circular diameter units"])
            particle.set_axes_lengths([p_group.attrs["Major axis length"],p_group.attrs["Minor axis length"]],p_group.attrs["Major axis length units"])
            particle.set_eccentricity(p_group.attrs["Eccentricity"])
            particle.set_intensity(p_group.attrs["Intensity"])
            #particle.set_zone(p_group.attrs["Zone"])
            
            particle.set_mask(np.array(p_group['Mask'][:]))
            
            if "Image" in p_group:
                particle.store_im(hs.signals.Signal2D(np.array(p_group['Image'][:])))
                
            p_list.append(particle)
    
    f.close()
    return(p_list)
