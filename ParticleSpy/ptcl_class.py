# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:51:58 2018

@author: qzo13262
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import interpolation
from ParticleSpy.particle_save import save_plist
from sklearn import feature_extraction, cluster, preprocessing
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
        
    def plot(self,prop_list=['area'],bins=20):
        """
        Plots properties of all particles in the Particle_list.
        
        If one property given, displays a histogram of the chosen particle property.
        
        If two properties given, displays a scatter plot of the two properties.
        
        Parameters
        ----------
        prop_list : str or list
            A particle property or a list of the names of the properties to plot.
        bins : int
            The number of bins in the histogram if plotting one property.
            
        Examples
        --------
        
        particles.plot('area', bins=20)
        
        particles.plot(['equivalent circular diameter','circularity'])
        
        """
        
        if isinstance(prop_list,str):
            self._plot_one_property(prop_list,bins)
        else:
            if len(prop_list) == 1:
                self._plot_one_property(prop_list[0],bins)
            elif len(prop_list) == 2:
                self._plot_two_properties(prop_list)
            else:
                print("Can only plot one or two properties, please change the length of the property list.")
        
        plt.show()
        
    def _plot_one_property(self,prop,bins):
        property_list = []
        
        for p in self.list:
            property_list.append(p.properties[prop]['value'])
            
        plt.hist(property_list,bins=bins,alpha=0.5)
        
        if self.list[0].properties[prop]['units'] == None:
            plt.xlabel(prop.capitalize())
        else:
            plt.xlabel(prop.capitalize()+" ("+self.list[0].properties[prop]['units']+")")
        plt.ylabel("No. of particles")
        
    def _plot_two_properties(self,prop_list):
        
        property_list_one = []
        property_list_two = []
        
        for p in self.list:
            property_list_one.append(p.properties[prop_list[0]]['value'])
            property_list_two.append(p.properties[prop_list[1]]['value'])
            
        plt.scatter(property_list_one,property_list_two,alpha=0.5)
        
        if self.list[0].properties[prop_list[0]]['units'] == None:
            plt.xlabel(prop_list[0].capitalize())
        else:
            plt.xlabel(prop_list[0].capitalize()+" ("+self.list[0].properties[prop_list[0]]['units']+")")
        
        if self.list[0].properties[prop_list[1]]['units'] == None:
            plt.ylabel(prop_list[1].capitalize())
        else:
            plt.ylabel(prop_list[1].capitalize()+" ("+self.list[0].properties[prop_list[1]]['units']+")")
        
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
            
    def show(self, params='Image', output=False):
        """
        display all particle images or other parameters

        Parameters
        ----------
        params : str, optional
            DESCRIPTION. The default is ['Image'].
            'Image'
            'maps'
            'area'
            'circularity'
        """
        self.normalize_boxing()
        
        num = len(self.list)
        cols = int(np.ceil(np.sqrt(num)))
        data_ls = []
        for index in range(num):
            if params == 'Image':
                data_ls.append(self.list[index].image.data)
            elif params == 'mask':
                data_ls.append(self.mask)
        self._show_images(data_ls, params, cols, np.arange(num))
        if output:
            return data_ls
            
    def _show_images(self, images, main_title, cols=1, titles=None):
        """
        Display a list of images in a single figure with matplotlib.
        
        Parameters
        ---------
        images: List of np.arrays compatible with plt.imshow.
        
        cols (Default = 1): Number of columns in figure (number of rows is 
                            set to np.ceil(n_images/float(cols))).
        
        titles: List of titles corresponding to each image. Must have
                the same length as titles.
        
        main_title: str. Name of the showing properties.
                    Can be 'Image', 'mask', 'area', etc.
        ---------
        Origin https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
        """
        assert((titles is None) or (len(images) == len(titles)))
        n_images = len(images)
        if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images+1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n+1)
            if image.ndim == 2:
                plt.gray()
            plt.axis('off')
            plt.imshow(image)
            a.set_title(title, fontsize=30)
        fig.set_size_inches(np.array([1,1]) * n_images)
        plt.title(main_title)
        plt.show()        
    
    
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
    vectorized = vec.fit_transform(properties_list).toarray()
        
    return(vec,vectorized)
