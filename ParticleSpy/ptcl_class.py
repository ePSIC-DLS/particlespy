# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:51:58 2018

@author: qzo13262
"""

import matplotlib.pyplot as plt
from ParticleSpy.particle_save import save_plist

class Particle(object):
    """A segmented particle object."""
    
    def set_origin(self, origin):
        """A container for the origin of data (filename, acquisition number etc.)"""
        self.origin = origin
        
    def set_area(self, area, units):
        self.area = area
        self.area_units = units
        
    def set_circularity(self, circularity):
        self.circularity = circularity
        
    def set_zone(self, zone):
        self.zone = zone
        
    def set_mask(self, mask):
        self.mask = mask
        
    def store_im(self,p_im):
        self.image = p_im
        
    def maps_gen(self):
        self.maps = {}
        
    def store_map(self,p_map,element):
        self.maps[element] = p_map
        
    def store_spectrum(self,spectrum,stype):
        self.spectrum = {}
        self.spectrum[stype] = spectrum
        
    def store_composition(self,composition):
        self.composition = {el.metadata.Sample.elements[0]:el.data for el in composition}
        
class Particle_list(object):
    """A particle list object."""
    
    def __init__(self):
        self.list = []
    
    def append(self,particle):
        self.list.append(particle)
        
    def save(self,filename):
        save_plist(self,filename)
        
    def plot_area(self):
        """
        Displays a plot of particle areas for analysed particles.
        """
        
        areas = []
        
        for p in self.list:
            areas.append(p.area)
            
        plt.hist(areas)
        plt.xlabel("Area ("+self.list[0].area_units+")")
        plt.ylabel("No. of particles")
        
    def plot_circularity(self):
        """
        Displays a plot of particle circularity for analysed particles.
        """
        
        circularities = []
        
        for p in self.list:
            circularities.append(p.circularity)
            
        plt.hist(circularities)
        plt.xlabel("Circularity")
        plt.ylabel("No. of particles")