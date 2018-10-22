# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:51:58 2018

@author: qzo13262
"""

class Particle(object):
    """A segmented particle object."""
    
    def set_origin(self, origin):
        """A container for the origin of data (filename, acquisition number etc.)"""
        self.origin = origin
        
    def set_area(self, area, units):
        self.area = area
        self.area_units = units
        
    def set_zone(self, zone):
        self.zone = zone
        
    def store_im(self,p_im):
        self.image = p_im