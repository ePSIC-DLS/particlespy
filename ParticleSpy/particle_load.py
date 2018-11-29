# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 17:11:22 2018

@author: qzo13262
"""

import h5py
from ParticleSpy.ptcl_class import Particle, Particle_list
import hyperspy as hs
import numpy as np

def load_plist(filename):
    f = h5py.File(filename,'r')
    p_list = Particle_list()
    
    for p_name in list(f.keys()):
        if p_name[:8] == 'Particle':
            p_group = f[p_name]
            particle = Particle()
            
            particle.set_area(p_group.attrs['Area'],p_group.attrs['Area units'])
            particle.set_circularity(p_group.attrs["Circularity"])
            #particle.set_zone(p_group.attrs["Zone"])
            
            particle.set_mask(np.array(p_group['Mask'][:]))
            
            if "Image" in p_group:
                particle.store_im(hs.signals.Signal2D(np.array(p_group['Image'][:])))
                
            p_list.append(particle)
    
    f.close()
    return(p_list)