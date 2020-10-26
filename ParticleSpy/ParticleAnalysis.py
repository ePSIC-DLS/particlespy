# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:35:23 2018

@author: qzo13262
"""

from ParticleSpy.segptcls import process
import numpy as np
from ParticleSpy.ptcl_class import Particle, Particle_list
from skimage import filters, morphology
from skimage.measure import label, regionprops, perimeter
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, KMeans
import ParticleSpy.find_zoneaxis as zone
import warnings
import h5py
import inspect
import pandas as pd
import trackpy
import os

def ParticleAnalysis(acquisition,parameters,particles=None,mask=np.zeros((1))):
    """
    Perform segmentation and analysis of images of particles.
    
    Parameters
    ----------
    acquisition: Hyperpsy signal object or list of hyperspy signal objects.
        Hyperpsy signal object containing a nanoparticle image or a list of signal
         objects that contains an image at position 0 and other datasets following.
    parameters: Dictionary of parameters
        The parameters can be input manually in to a dictionary or can be generated
        using param_generator().
    particles: list
        List of already analysed particles that the output can be appended
        to.
    mask: Numpy array
        Numpy array of same 2D size as acquisition that contains a mask of presegmented
        particles.
        
    Returns
    -------
    Particle_list object
    """
    
    if particles==None:
        particles=Particle_list()
    
    #Check if input is list of signal objects or single one
    if isinstance(acquisition,list):
        image = acquisition[0]
    else:
        image = acquisition
    
    if str(mask) == 'UI':
        labeled = label(np.load(os.path.dirname(inspect.getfile(process))+'/Parameters/manual_mask.npy'))
        #plt.imshow(labeled)
        #morphology.remove_small_objects(labeled,30,in_place=True)
    elif mask.sum()==0:
        labeled = process(image,parameters)
        #plt.imshow(labeled,cmap=plt.cm.nipy_spectral)
        #labels = np.unique(labeled).tolist() #some labeled number have been removed by "remove_small_holes" function
    else:
        labeled = label(mask)
        
    for region in regionprops(labeled): #'count' start with 1, 0 is background
        p = Particle()
        
        p_im = np.zeros_like(image.data)
        p_im[labeled==region.label] = image.data[labeled==region.label] - np.min(image.data[labeled==region.label])
        
        maskp = np.zeros_like(image.data)
        maskp[labeled==region.label] = 1
        
        #Calculate average background around image
        dilated_mask = morphology.binary_dilation(maskp).astype(int)
        dilated_mask2 = morphology.binary_dilation(dilated_mask).astype(int)
        boundary_mask = dilated_mask2 - dilated_mask
        p.background = np.sum(boundary_mask*image.data)/np.count_nonzero(boundary_mask)
        
        #origin = ac_number
        #p.set_origin(origin)
        
        #Set area
        cal_area = region.area*image.axes_manager[0].scale*image.axes_manager[1].scale
        try:
            area_units = image.axes_manager[0].units + "^2"
        except:
            area_units = "Unknown Units^2"

        p.set_area(cal_area,area_units)
        
        #Set diam measures
        cal_circdiam = 2*(cal_area**0.5)/np.pi
        diam_units = image.axes_manager[0].units
        p.set_circdiam(cal_circdiam,diam_units)
        
        #Set particle x, y coordinates
        p.set_property("x",region.centroid[0]*image.axes_manager[0].scale,image.axes_manager[0].units)
        p.set_property("y",region.centroid[1]*image.axes_manager[1].scale,image.axes_manager[1].units)
        
        cal_axes_lengths = (region.major_axis_length*image.axes_manager[0].scale,region.minor_axis_length*image.axes_manager[0].scale)
        #Note: the above only works for square pixels
        p.set_axes_lengths(cal_axes_lengths,diam_units)
        
        #Set shape measures
        peri = image.axes_manager[0].scale*perimeter(maskp,neighbourhood=4)
        circularity = 4*3.14159265*p.properties['area']['value']/(peri**2)
        p.set_circularity(circularity)
        eccentricity = region.eccentricity
        p.set_eccentricity(eccentricity)
        p.set_property("solidity",region.solidity,None)
        
        #Set total image intensity
        intensity = ((image.data - p.background)*maskp).sum()
        p.set_intensity(intensity)
        p.set_property("intensity_max",((image.data - p.background)*maskp).max(),None)
        p.set_property("intensity_std",((image.data - p.background)*maskp).std()/p.properties['intensity_max']['value'],None)
        
        #Set zoneaxis
        '''im_smooth = filters.gaussian(np.uint16(p_im),1)
        im_zone = np.zeros_like(im_smooth)
        im_zone[im_smooth>0] = im_smooth[im_smooth>0] - im_smooth[im_smooth>0].mean()
        im_zone[im_zone<0] = 0
        p.set_zone(zone.find_zoneaxis(im_zone))
        if p.zone != None:
            plt.imshow(im_zone)
            plt.show()'''
        
        #Set mask
        p.set_mask(maskp)
        
        p.set_property('frame',None,None)
        
        if parameters.store["store_im"]==True:
            store_image(p,image,parameters)
            
        if isinstance(acquisition,list):
            p.spectrum = {}
            for ac in acquisition[1:]:
                if ac.metadata.Signal.signal_type == 'EDS_TEM':
                    ac.set_elements(parameters.eds['elements'])
                    ac.add_lines()
                    store_spectrum(p,ac,'EDS_TEM')
                    if parameters.store["store_maps"]:
                        store_maps(p,ac,parameters)
                    if parameters.eds["factors"]!=False:
                        get_composition(p,parameters)
                elif ac.metadata.Signal.signal_type == 'EELS':
                    if 'high-loss' in ac.metadata.General.title:
                        store_spectrum(p,ac,'EELS-HL')
                    elif 'low-loss' in ac.metadata.General.title:
                        store_spectrum(p,ac,'EELS-LL')
                    else:
                        store_spectrum(p,ac,ac.metadata.Signal.signal_type)
                else:
                    if ac.metadata.Signal.signal_type:
                        store_spectrum(p,ac,ac.metadata.Signal.signal_type)
                    else:
                        warnings.warn("You have input data that does not have a defined signal type and therefore will not be processed."+
                              " You need to define signal_type in the metadata for anything other than the first dataset.")
        
        particles.append(p)
        
    return(particles)
    
def ParticleAnalysisSeries(image_series,parameters,particles=None):
    """
    Perform segmentation and analysis of times series of particles.
    
    Parameters
    ----------
    image_series: Hyperspy signal object or list of hyperspy signal objects.
        Hyperpsy signal object containing nanoparticle images or a list of signal
         objects that contains a time series.
    parameters: Dictionary of parameters
        The parameters can be input manually in to a dictionary or can be generated
        using param_generator().
    particles: list
        List of already analysed particles that the output can be appended
        to.

    Returns
    -------
    Particle_list object
    """
    
    particles = Particle_list()
    if isinstance(image_series,list):
        for i,image in enumerate(image_series):
            ParticleAnalysis(image,parameters,particles)
            for particle in particles.list:
                if particle.properties['frame']['value'] == None:
                    particle.set_property('frame',i,None)
    else:
        for i,image in enumerate(image_series.inav):
            ParticleAnalysis(image,parameters,particles)
            for particle in particles.list:
                if particle.properties['frame']['value'] == None:
                    particle.set_property('frame',i,None)
    
    return(particles)

def timeseriesanalysis(particles,max_dist=1,memory=3,properties=['area']):
    """
    Perform tracking of particles for times series data.

    Parameters
    ----------
    particles : Particle_list object.
    max_dist : int
        The maximum distance between the same particle in subsequent images.
    memory : int
        The number of frames to remember particles over.
    properties : list
        A list of particle properties to track over the time series.

    Returns
    -------
    Pandas DataFrame of tracjectories.

    """
    df = pd.DataFrame(columns=['y','x']+properties+['frame'])
    for particle in particles.list:
        pd_dict = {'x':particle.properties['x']['value'],
                   'y':particle.properties['y']['value']}
        for property in properties:
            pd_dict.update({property:particle.properties[property]['value']})
        pd_dict.update({'frame':particle.properties['frame']['value']})
        df = df.append([pd_dict])
        
    t = trackpy.link(df,max_dist,memory=memory)
    return(t)

def CreateFeatures(image, intensity = True, edges = True, texture = True, sigma = 1, high_sigma = 16, disk_size = 20):

    shape = [image.shape[0], image.shape[1], 1]

    selem = morphology.disk(disk_size)

    image_stack = np.zeros(shape)
    im_blur = filters.gaussian(image, sigma)

    if intensity:
        new_layer = np.reshape(filters.gaussian(image, sigma), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

        new_layer = np.reshape(filters.difference_of_gaussians(image,low_sigma= sigma, high_sigma=high_sigma), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)
        """
        new_layer = np.reshape(filters.median(image,selem), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

        new_layer = np.reshape(filters.rank.minimum(image,selem), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

        new_layer = np.reshape(filters.rank.maximum(image, selem), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)
        """
    if edges:
        new_layer = np.reshape(filters.sobel(im_blur), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

    if texture:
        new_layer = np.reshape(filters.hessian(im_blur, mode='constant',), shape)
        image_stack = np.append(image_stack, new_layer, axis=2)


    return image_stack

def ClusterLearnOld(image, method='KMeans',
                parameters=[{'kernel': 'gaussian', 'sigma': 1}, 
                            {'kernel': 'sobel'},
                            {'kernel': 'hessian', 'black ridges': False},
                            {'kernel': 'rank mean', 'disk size': 20},
                            {'kernel': 'median', 'disk size': 20},
                            {'kernel': 'minimum', 'disk size': 20},
                            {'kernel': 'maximum', 'disk size': 20},
                            {'kernel': 'bilateral', 'disk size': 20},
                            {'kernel': 'entropy', 'disk size': 20},
                            {'kernel': 'gabor', 'frequency': 100},
                            {'kernel': 'gaussian diff', 'low sigma': 1, 'high sigma': 5}]):
    """
    Creates masks of given images using scikit learn clustering methods, or by using provided training data.
    
    Parameters
    ----------
    image: Hyperspy signal object or list of hyperspy signal objects.
        Hyperpsy signal object containing nanoparticle images
    method: Clustering algorithm used to generate mask.
    disk_size: Size of the local pixel neighbourhood considered by select 
        segmentation methods.
    parameters: List of dictionaries of Parameters for segmentation methods used in clustering
        The parameters can be inputted manually or use the default .

    Returns
    -------
    generated mask
    """

    image = image.data
    image = preprocessing.maxabs_scale(image)
    shape = [image.shape[0], image.shape[1], 1]

    image_stack = np.zeros(shape)

    for i in range(len(parameters)):
        if parameters[i]['kernel'] == 'gaussian':
            sigma = parameters[i]['sigma']
            new_layer = np.reshape(filters.gaussian(image, sigma), shape)
        elif parameters[i]['kernel'] == 'gaussian diff':
            low_sigma = parameters[i]['low sigma']
            high_sigma = parameters[i]['high sigma']
            new_layer = np.reshape(filters.difference_of_gaussians(image, low_sigma, high_sigma), shape)
        elif parameters[i]['kernel'] == 'sobel':
            new_layer = np.reshape(filters.sobel(image), shape)
        elif parameters[i]['kernel'] == 'hessian':
            black_ridges = parameters[i]['black ridges']
            new_layer = np.reshape(filters.hessian(image, black_ridges, mode='constant',), shape)
        elif parameters[i]['kernel'] == 'rank mean':
            selem = morphology.disk(parameters[i]['disk size'])
            new_layer = np.reshape(filters.rank.mean(image,selem), shape)
        elif parameters[i]['kernel'] == 'median':
            selem = morphology.disk(parameters[i]['disk size'])
            new_layer = np.reshape(filters.median(image,selem), shape)
        elif parameters[i]['kernel'] == 'min':
            selem = morphology.disk(parameters[i]['disk size'])
            new_layer = np.reshape(filters.rank.minimum(image,selem), shape)
        elif parameters[i]['kernel'] == 'max':
            selem = morphology.disk(parameters[i]['disk size'])
            new_layer = np.reshape(filters.rank.maximum(image, selem), shape)
        elif parameters[i]['kernel'] == 'bilateral':
            selem = morphology.disk(parameters[i]['disk size'])
            new_layer = np.reshape(filters.rank.mean_bilateral(image, selem), shape)
        elif parameters[i]['kernel'] == 'entropy':
            selem = morphology.disk(parameters[i]['disk size'])
            new_layer = np.reshape(filters.rank.entropy(image, selem), shape)
        elif parameters[i]['kernel'] == 'gabor':
            freq = parameters[i]['frequency']
            new_layer, _ = filters.gabor(image,freq)
            new_layer = np.reshape(new_layer, shape)
        image_stack = np.append(image_stack, new_layer, axis=2)

    pixel_stacks = np.zeros([shape[0]*shape[1],image_stack.shape[2]])
    for i in range(shape[1]):
        pixel_stacks[i*shape[0]:(i+1)*shape[0],:] = image_stack[:,i,:]
        
    pixel_stacks = preprocessing.scale(pixel_stacks)

    if method == 'KMeans':
        labels = KMeans(n_clusters=2,init='random',n_init=10).fit_predict(pixel_stacks)
    elif method == 'DBscan':
        labels = DBSCAN().fit_predict(pixel_stacks)
        
    mask = np.zeros_like(image)
    for i in range(shape[1]):
        mask[:,i] = labels[i*shape[0]:(i+1)*shape[0]]
    
    return mask

def ClusterLearn(image, method='KMeans'):
    """
    Creates masks of given images using scikit learn clustering methods.
    
    Parameters
    ----------
    image: Hyperspy signal object or list of hyperspy signal objects.
        Hyperpsy signal object containing nanoparticle images
    method: Clustering algorithm used to generate mask.
    disk_size: Size of the local pixel neighbourhood considered by select 
        segmentation methods.
    parameters: List of dictionaries of Parameters for segmentation methods used in clustering
        The parameters can be inputted manually or use the default .

    Returns
    -------
    generated mask
    """

    image = image.data
    image = preprocessing.maxabs_scale(image)
    shape = [image.shape[0], image.shape[1], 1]

    image_stack = CreateFeatures(image)

    pixel_stacks = np.zeros([shape[0]*shape[1],image_stack.shape[2]])
    for i in range(shape[1]):
        pixel_stacks[i*shape[0]:(i+1)*shape[0],:] = image_stack[:,i,:]
        
    pixel_stacks = preprocessing.scale(pixel_stacks)

    if method == 'KMeans':
        labels = KMeans(n_clusters=2,init='random',n_init=10).fit_predict(pixel_stacks)
    elif method == 'DBscan':
        labels = DBSCAN().fit_predict(pixel_stacks)
        
    mask = np.zeros_like(image)
    for i in range(shape[1]):
        mask[:,i] = labels[i*shape[0]:(i+1)*shape[0]]
    
    return mask

def ClusterLearnSeries(image_set, method='KMeans', parameters=[{'kernel': 'gaussian', 'sigma': 1}, 
                                                    {'kernel': 'sobel'},
                                                    {'kernel': 'hessian', 'black ridges': False},
                                                    {'kernel': 'rank mean', 'disk size': 20},
                                                    {'kernel': 'median', 'disk size': 20},
                                                    {'kernel': 'minimum', 'disk size': 20},
                                                    {'kernel': 'maximum', 'disk size': 20},
                                                    {'kernel': 'bilateral', 'disk size': 20},
                                                    {'kernel': 'entropy', 'disk size': 20},
                                                    {'kernel': 'gabor', 'frequency': 100},
                                                    {'kernel': 'gaussian diff', 'low sigma': 1, 'high sigma': None}]):
    """
    Creates masks of sets of images using scikit learn clustering methods.
    
    Parameters
    ----------
    image_set: list of hyperspy signal objects.
        List of Hyperpsy signal object containing nanoparticle images
    method: Clustering algorithm used to generate mask.
    disk_size: Size of the local pixel neighbourhood considered by select 
        segmentation methods.
    parameters: List of dictionaries of Parameters for segmentation methods used in clustering
        The parameters can be input manually in to a dictionary or can be generated
        using param_generator().

    Returns
    -------
    list of generated mask per image
    """

    mask_set = []
    for image in image_set:
        mask_set.append(ClusterLearn(image,method))
    
    return mask_set

def ClusterTrained(image, labels, classifier):

    labels = labels.astype(np.float64)
    shape = image.data.shape
    image = image.data

    features = CreateFeatures(image)
    features = np.rot90(np.rot90(features, axes=(2,0)), axes=(1,2))
    #features are num/x/y

    thin_mask = np.zeros([shape[0],shape[1]])

    c = 1
    for i in range(0,3):
        non_zero = (labels[:,:,i] != 0)
        if np.any(non_zero) == True:
            thin_mask += c*non_zero.astype(int)
            c += 1

    training_data = features[:, thin_mask > 0].T
    #training data is number of labeled pixels by number of features
    training_labels = thin_mask[thin_mask > 0].ravel()
    training_labels = training_labels.astype('int')
    #training labels is labelled pixels in 1D array

    classifier.fit(training_data, training_labels)
    print('finish training')
    #train classifier on  labelled data
    data = features[:, thin_mask == 0].T
    #unlabelled data
    pred_labels = classifier.predict(data)
    #predict labels for rest of image

    output = np.copy(thin_mask)
    output[thin_mask == 0] = pred_labels

    return output, classifier

def store_image(particle,image,params):
    ii = np.where(particle.mask)
            
    box_x_min = np.min(ii[0])
    box_x_max = np.max(ii[0])
    box_y_max = np.max(ii[1])
    box_y_min = np.min(ii[1])
    pad = params.store['pad']
    
    if params.store['p_only']==True:
        image = image*particle.mask
    
    if box_y_min-pad > 0 and box_x_min-pad > 0 and box_x_max+pad < particle.mask.shape[0] and box_y_max+pad < particle.mask.shape[1]:
        p_boxed = image.isig[(box_y_min-pad):(box_y_max+pad),(box_x_min-pad):(box_x_max+pad)]
    else:
        p_boxed = image.isig[(box_y_min):(box_y_max),(box_x_min):(box_x_max)]
    particle.store_im(p_boxed)
    
def store_maps(particle,ac,params):
    maps = ac.get_lines_intensity()
    particle.maps_gen()
    
    for el_map in maps:
        ii = np.where(particle.mask)
                
        box_x_min = np.min(ii[0])
        box_x_max = np.max(ii[0])
        box_y_max = np.max(ii[1])
        box_y_min = np.min(ii[1])
        pad = params.store['pad']
        
        if box_y_min-pad > 0 and box_x_min-pad > 0 and box_x_max+pad < particle.mask.shape[0] and box_y_max+pad < particle.mask.shape[1]:
            p_boxed = el_map.inav[(box_y_min-pad):(box_y_max+pad),(box_x_min-pad):(box_x_max+pad)]
        else:
            p_boxed = el_map.inav[(box_y_min):(box_y_max),(box_x_min):(box_x_max)]
        particle.store_map(p_boxed,p_boxed.metadata.Sample.elements[0])
        
def store_spectrum(particle,ac,stype):
    ac_particle = ac.transpose()*particle.mask
    ac_particle = ac_particle.transpose()
    ac_particle_spectrum = ac_particle.sum()
    if '-' in stype:
        ac_particle_spectrum.set_signal_type(stype.rpartition('-')[0])
    else:
        ac_particle_spectrum.set_signal_type(stype)
    particle.store_spectrum(ac_particle_spectrum,stype)
        
def get_composition(particle,params):
    bw = particle.spectrum['EDS_TEM'].estimate_background_windows(line_width=[5.0, 2.0])
    intensities = particle.spectrum['EDS_TEM'].get_lines_intensity(background_windows=bw)
    atomic_percent = particle.spectrum['EDS_TEM'].quantification(intensities, method=params.eds['method'],factors=params.eds['factors'])
    particle.store_composition(atomic_percent)
    
class parameters(object):
    """A parameters object."""
    
    def generate(self,threshold='otsu',watershed=False,watershed_size=0,
                 watershed_erosion=0,invert=False,min_size=0,store_im=False,
                 pad=5,rb_kernel=0,gaussian=0,local_size=1):
        self.segment = {}
        self.segment['threshold'] = threshold
        self.segment['watershed'] = watershed
        self.segment['watershed_size'] = watershed_size
        self.segment['watershed_erosion'] = watershed_erosion
        self.segment['invert'] = invert
        self.segment['min_size'] = min_size
        self.segment['rb_kernel'] = rb_kernel
        self.segment['gaussian'] = gaussian
        self.segment['local_size'] = local_size
        
        self.store = {}
        self.store['store_im'] = store_im
        self.store['pad'] = pad
        self.store['p_only'] = False
        
        self.generate_eds()
        
    def generate_eds(self,eds_method=False,elements=False, factors=False,
                     store_maps=False):
        self.eds = {}
        self.eds['method'] = eds_method
        self.eds['elements'] = elements
        self.eds['factors'] = factors
        
        self.store['store_maps'] = store_maps
    
    def save(self,filename=os.path.dirname(inspect.getfile(process))+'/Parameters/parameters_current.hdf5'):
        f = h5py.File(filename,'w')
        
        segment = f.create_group("segment")
        store = f.create_group("store")
        eds = f.create_group("eds")
        
        segment.attrs["threshold"] = self.segment['threshold']
        segment.attrs["watershed"] = self.segment['watershed']
        segment.attrs["watershed_size"] = self.segment['watershed_size']
        segment.attrs["watershed_erosion"] = self.segment['watershed_erosion']
        segment.attrs["invert"] = self.segment['invert']
        segment.attrs["min_size"] = self.segment['min_size']
        segment.attrs["rb_kernel"] = self.segment['rb_kernel']
        segment.attrs["gaussian"] = self.segment['gaussian']
        segment.attrs["local_size"] = self.segment['local_size']
        store.attrs['store_im'] = self.store['store_im']
        store.attrs['pad'] = self.store['pad']
        store.attrs['store_maps'] = self.store['store_maps']
        store.attrs['p_only'] = self.store['p_only']
        eds.attrs['method'] = self.eds['method']
        eds.attrs['elements'] = self.eds['elements']
        eds.attrs['factors'] = self.eds['factors']
        
        f.close()
        
    def load(self,filename=os.path.dirname(inspect.getfile(process))+'/Parameters/parameters_current.hdf5'):
        f = h5py.File(filename,'r')
        
        segment = f["segment"]
        store = f["store"]
        eds = f["eds"]
        
        self.segment = {}
        self.store = {}
        self.eds = {}
        
        self.segment['threshold'] = segment.attrs["threshold"]
        self.segment['watershed'] = segment.attrs["watershed"]
        self.segment['watershed_size'] = segment.attrs["watershed_size"]
        self.segment['watershed_erosion'] = segment.attrs["watershed_erosion"]
        self.segment['invert'] = segment.attrs["invert"]
        self.segment['min_size'] = segment.attrs["min_size"]
        self.segment['rb_kernel'] = segment.attrs["rb_kernel"]
        self.segment['gaussian'] = segment.attrs["gaussian"]
        self.segment['local_size'] = segment.attrs["local_size"]
        self.store['store_im'] = store.attrs['store_im']
        self.store['pad'] = store.attrs['pad']
        self.store['store_maps'] = store.attrs['store_maps']
        self.store['p_only'] = store.attrs['p_only']
        self.eds['method'] = eds.attrs['method']
        self.eds['elements'] = eds.attrs['elements']
        self.eds['factors'] = eds.attrs['factors']
        
        f.close()
