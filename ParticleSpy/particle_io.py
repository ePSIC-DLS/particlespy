# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:23:59 2018

@author: qzo13262
"""

import h5py
import ptcl_class
import hyperspy as hs
import numpy as np

def save_plist(p_list,filename):
    f = h5py.File(filename,'w')
    
    for i, particle in enumerate(p_list.list):
        p_group = f.create_group("Particle "+str(i))
        
        p_group.attrs["Area"] = particle.area
        p_group.attrs["Area units"] = particle.area_units
        p_group.attrs["Circularity"] = particle.circularity
        #p_group.attrs["Zone"] = particle.zone
        
        p_group.create_dataset("Mask",data=particle.mask)
        
        if hasattr(particle, 'image'):
            p_group.create_dataset("Image",data=particle.image.data)
        
    f.close()
        
def load_plist(filename):
    f = h5py.File(filename,'r')
    p_list = ptcl_class.Particle_list()
    
    for p_name in list(f.keys()):
        if p_name[:8] == 'Particle':
            p_group = f[p_name]
            particle = ptcl_class.Particle()
            
            particle.set_area(p_group.attrs['Area'],p_group.attrs['Area units'])
            particle.set_circularity(p_group.attrs["Circularity"])
            #particle.set_zone(p_group.attrs["Zone"])
            
            particle.set_mask(np.array(p_group['Mask'][:]))
            
            if "Image" in p_group:
                particle.store_im(hs.signals.Signal2D(np.array(p_group['Image'][:])))
                
            p_list.append(particle)
    
    f.close()
    return(p_list)
