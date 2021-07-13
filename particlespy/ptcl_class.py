# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:51:58 2018

@author: qzo13262
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import interpolation
from particlespy.particle_save import save_plist
from sklearn import feature_extraction, cluster, preprocessing
import itertools as it
from mpl_toolkits.mplot3d import Axes3D

class particle(object):
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
        Give a particle() object an arbitrary property.
        
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
        
    def set_boundingbox(self, bbox):
        self.bbox = bbox
        
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
        
class particle_list(object):
    """A particle list object."""
    
    def __init__(self):
        self.list = []
    
    def append(self,particle):
        self.list.append(particle)
        
    def save(self,filename):
        save_plist(self,filename)

    def plot(self,prop_list=['area'],**kwargs):
        """
        Plots properties of all particles in the Particle_list.
        
        If one property given, displays a histogram of the chosen particle property.
        
        If two properties given, displays a scatter plot of the two properties.
        
        Parameters
        ----------
        prop_list : str or list
            A particle property or a list of the names of the properties to plot.
        **kwargs
            Keyword arguments for matplotlib plotting functions.
            
        Examples
        --------
        
        particles.plot('area', bins=20)
        
        particles.plot(['equivalent circular diameter','circularity'])
        
        """
        
        if isinstance(prop_list,str):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            self._plot_one_property(prop_list,ax,**kwargs)
        else:
            if len(prop_list) == 1:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                self._plot_one_property(prop_list[0],ax,**kwargs)
            elif len(prop_list) == 2:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                self._plot_two_properties(prop_list,ax,**kwargs)
            elif len(prop_list) == 3:
                fig3d = plt.figure()
                ax = fig3d.add_subplot(111, projection='3d')
                self._plot_three_properties(prop_list,ax,**kwargs)
            else:
                print("Can only plot a maximum of three properties, please change the length of the property list.")
        
        plt.show()
        
    def _plot_one_property(self,prop,ax,**kwargs):
        property_list = []
        
        for p in self.list:
            property_list.append(p.properties[prop]['value'])
            
        ax.hist(property_list,**kwargs)
        
        if self.list[0].properties[prop]['units'] == None:
            ax.set_xlabel(prop.capitalize())
        else:
            ax.set_xlabel(prop.capitalize()+" ("+self.list[0].properties[prop]['units']+")")
        plt.ylabel("No. of particles")
        
    def _plot_two_properties(self,prop_list,ax,**kwargs):
        
        property_list_one = []
        property_list_two = []
        
        for p in self.list:
            property_list_one.append(p.properties[prop_list[0]]['value'])
            property_list_two.append(p.properties[prop_list[1]]['value'])
        
        ax.scatter(property_list_one,property_list_two,**kwargs)
        
        if self.list[0].properties[prop_list[0]]['units'] == None:
            ax.set_xlabel(prop_list[0].capitalize())
        else:
            ax.set_xlabel(prop_list[0].capitalize()+" ("+self.list[0].properties[prop_list[0]]['units']+")")
        
        if self.list[0].properties[prop_list[1]]['units'] == None:
            ax.set_ylabel(prop_list[1].capitalize())
        else:
            ax.set_ylabel(prop_list[1].capitalize()+" ("+self.list[0].properties[prop_list[1]]['units']+")")

    def _plot_three_properties(self, prop_list,ax,**kwargs):

        property_list_one = []
        property_list_two = []
        property_list_three = []
        
        for p in self.list:
            property_list_one.append(p.properties[prop_list[0]]['value'])
            property_list_two.append(p.properties[prop_list[1]]['value'])
            property_list_three.append(p.properties[prop_list[2]]['value'])

        ax.scatter(property_list_one, property_list_two, property_list_three,**kwargs)

        if self.list[0].properties[prop_list[0]]['units'] == None:
            ax.set_xlabel(prop_list[0].capitalize())
        else:
            ax.set_xlabel(prop_list[0].capitalize()+" ("+self.list[0].properties[prop_list[0]]['units']+")")

        if self.list[0].properties[prop_list[1]]['units'] == None:
            ax.set_ylabel(prop_list[1].capitalize())
        else:
            ax.set_ylabel(prop_list[1].capitalize()+" ("+self.list[0].properties[prop_list[1]]['units']+")")

        if self.list[0].properties[prop_list[2]]['units'] == None:
            ax.set_zlabel(prop_list[2].capitalize())
        else:
            ax.set_zlabel(prop_list[2].capitalize()+" ("+self.list[0].properties[prop_list[2]]['units']+")")
        
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
            
    def cluster_particles(self,algorithm='Kmeans',properties=None,n_clusters=2,eps=0.2,min_samples=5):
        """
        Cluster particles in to different populations based on specified properties.
        
        Parameters
        ----------
        algorithm: str
            The algorithm to use for clustering.
            Options are 'Kmeans','DBSCAN','OPTICS','AffinityPropagation'.
        properties: list
            A list of the properties upon which to base the clustering.
        n_clusters: int
            The number of clusters to split the data into.
            Used for Kmeans.
        eps: float
            The distance between samples.
            Used for DBSCAN.
        min_samples: int
            The minimum number of samples within the eps distance to be classed as a cluster.
            Used for DBSCAN and OPTICS.
        
        Returns
        -------
        List of Particle_list() objects.
        """
        vec,feature_array = _extract_features(self,properties)
        
        feature_array = preprocessing.scale(feature_array)
        
        if algorithm=='Kmeans':
            cluster_out = cluster.KMeans(n_clusters=n_clusters).fit_predict(feature_array)
            start = 0
        elif algorithm=='DBSCAN':
            cluster_out = cluster.DBSCAN(eps=eps,min_samples=min_samples).fit_predict(feature_array)
            start = -1
        elif algorithm=='OPTICS':
            cluster_out = cluster.OPTICS(min_samples=min_samples).fit_predict(feature_array)
            start = -1
        elif algorithm=='AffinityPropagation':
            cluster_out = cluster.AffinityPropagation().fit_predict(feature_array)
            start = 0
        
        for i,p in enumerate(self.list):
            p.cluster_number = cluster_out[i]
        
        plist_clusters = []
        for n in range(start,cluster_out.max()+1):
            p_list_new = particle_list()
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
    vectorized = vec.fit_transform(properties_list).toarray()
        
    return(vec,vectorized)
