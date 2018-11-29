# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:23:59 2018

@author: qzo13262
"""

import h5py

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
        

