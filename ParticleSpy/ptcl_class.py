# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:51:58 2018

@author: qzo13262
"""

import matplotlib.pyplot as plt
from ParticleSpy.particle_save import save_plist

class Particle(object):
    """A segmented particle object.
    
    Attributes
    ----------
    properties : dict
        Dictionary of particle properties created by the ParticleAnalysis() function.
    origin : str
        Origin of particle data, e.g. filename or acquisition number.
    zone : str
        Zone axis of particle.
    mask : array
        Boolean array corresponding to the particle pixels on the original image.
    image : Hyperspy signal object
        Image of particle.
    maps : dict
        Dictionary containing elemental maps of the particle.
    spectrum : Hyperspy signal object
        Spectrum obtained from the particle.
    composition : dict
        Dictionary of composition values for the particle.
    
    """
    
    def __init__(self):
        self.properties = {}
    
    def set_origin(self, origin):
        """A container for the origin of data (filename, acquisition number etc.)"""
        self.origin = origin
        
    def set_area(self, area, units):
        self.properties['area'] = {'value':area,'units':units}
        
    def set_circdiam(self, circdiam, units):
        self.properties['equivalent circular diameter'] = {'value':circdiam,'units':units}
        
    def set_axes_lengths(self,axeslengths,units):
        self.properties['major axis length'] = {'value':axeslengths[0],'units':units}
        self.properties['minor axis length'] = {'value':axeslengths[1],'units':units}
        
    def set_circularity(self, circularity):
        self.properties['circularity'] = {'value':circularity,'units':None}
        
    def set_eccentricity(self,eccentricity):
        self.properties['eccentricity'] = {'value':eccentricity,'units':None}
        
    def set_intensity(self,intensity):
        self.properties['intensity'] = {'value':intensity,'units':None}
        
    def set_property(self,propname,value,units):
        self.properties[propname] = {'value':value, 'units': units}
        
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
        
    def plot_area(self,bins=20):
        """
        Displays a plot of particle areas for analysed particles.
        """
        
        areas = []
        
        for p in self.list:
            areas.append(p.properties['area']['value'])
            
        plt.hist(areas,bins=bins)
        plt.xlabel("Area ("+self.list[0].properties['area']['units']+")")
        plt.ylabel("No. of particles")
        
    def plot_circularity(self,bins=20):
        """
        Displays a plot of particle circularity for analysed particles.
        """
        
        circularities = []
        
        for p in self.list:
            circularities.append(p.properties['circularity']['value'])
            
        plt.hist(circularities,bins=bins)
        plt.xlabel("Circularity")
        plt.ylabel("No. of particles")
        
    def plot(self,prop='area',bins=20):
        """
        Displays a histogram of the chosen particle property.
        
        Parameters
        ----------
        prop : str
            The name of the property to plot as a histogram.
        bins : int
            The number of bins in the histogram.
        
        """
        
        property_list = []
        
        for p in self.list:
            property_list.append(p.properties[prop]['value'])
            
        plt.hist(property_list,bins=bins)
        if self.list[0].properties[prop]['units'] == None:
            plt.xlabel(prop.capitalize())
        else:
            plt.xlabel(prop.capitalize()+" ("+self.list[0].properties[prop]['units']+")")
        plt.ylabel("No. of particles")