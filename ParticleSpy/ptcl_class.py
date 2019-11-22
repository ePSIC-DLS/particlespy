# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:51:58 2018

@author: qzo13262
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import interpolation
from ParticleSpy.particle_save import save_plist
from sklearn import feature_extraction, cluster
import itertools as it

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
        """
        Give a Particle() object an arbitrary property.
        
        Parameters
        ----------
        propname : str
            The name of the property to set.
        value : 
            The value of the property.
        units :
            The units of the property.
            
        Example
        -------
        >>> particle.set_property('area',10.0,'nm')
        """
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
        plt.show()
        
    def normalize_boxing(self,even=False):
        """
        Normalizes the size of all particle images so that their dimensions are
        equal.
        
        Example
        -------
        >>> particles.normalize_boxing()
        """
        
        dimensions = []
        for particle in self.list:
            dimensions.append(particle.image.data.shape[0])
            dimensions.append(particle.image.data.shape[1])
            
        if even==True:
            if (max(dimensions) % 2) == 0:
                max_dim = max(dimensions)
            else:
                max_dim = max(dimensions) + 1
        else:
            max_dim = max(dimensions)
        
        for i,particle in enumerate(self.list):
            x_diff = max_dim - particle.image.data.shape[0]
            y_diff = max_dim - particle.image.data.shape[1]
            
            minval = particle.image.data.min()
            
            new_im = np.full((max_dim,max_dim),minval)
            
            new_im[:particle.image.data.shape[0],
                   :particle.image.data.shape[1]] = particle.image.data
                   
            new_im = interpolation.shift(new_im,(x_diff/2,y_diff/2),cval=minval)
            
            particle.image.data = new_im
            
            particle.image.axes_manager[0].size = particle.image.data.shape[0]
            particle.image.axes_manager[1].size = particle.image.data.shape[1]
            
    def cluster_particles(self,algorithm='Kmeans',properties=None,n_clusters=2):
        feature_array = _extract_features(self,properties)
        
        if algorithm=='Kmeans':
            cluster_out = cluster.KMeans(n_clusters=n_clusters).fit_predict(feature_array)
            
        for i,p in enumerate(self.list):
            p.cluster_number = cluster_out[i]
        
        plist_clusters = []
        for n in n_clusters:
            p_list_new = Particle_list()
            p_list_new.list = list(it.compress(self.list,[c==n for c in cluster_out]))
            plist_clusters.append(p_list_new)
        
        return(plist_clusters)

def _extract_features(particles,properties=None):
    if properties==None:
        properties = particles.list[0].properties
    
    properties_list = []
    for particle in particles.list:
        properties_list.append({p:particle.properties[p]['value'] for p in properties})
        
    vec = feature_extraction.DictVectorizer()
    vectorized = vec.fit_transform(properties_list)
        
    return(vec,vectorized)
