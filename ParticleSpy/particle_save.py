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
        
        for prop in particle.properties:
            if particle.properties[prop]['value'] is not None:
                p_group.attrs[prop] = particle.properties[prop]['value']
            if particle.properties[prop]['units'] is not None:
                p_group.attrs[prop+' units'] = particle.properties[prop]['units']
        
        p_group.create_dataset("Mask",data=particle.mask)
        
        if hasattr(particle, 'image'):
            p_group.create_dataset("Image",data=particle.image.data)
        
    f.close()
