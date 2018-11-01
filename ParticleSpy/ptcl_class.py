# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:51:58 2018

@author: qzo13262
"""

import matplotlib.pyplot as plt

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
        
class Particle_list(object):
    """A particle list object."""
    
    def __init__(self):
        self.list = []
    
    def append(self,particle):
        self.list.append(particle)
        
    def plot_area(self):
        """
        Displays a plot of particle areas for analysed particles.
        """
        
        areas = []
        
        for p in self.list:
            areas.append(p.area)
            
        plt.hist(areas)
        plt.xlabel(self.list[0].area_units)
        plt.ylabel("No. of particles")