# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:35:23 2018

@author: qzo13262
"""

from ParticleSpy.segptcls import process
import numpy as np
from ParticleSpy.ptcl_class import Particle, Particle_list
from skimage import filters
from skimage.measure import label, regionprops, perimeter
import ParticleSpy.find_zoneaxis as zone
import warnings
import h5py
import inspect
import matplotlib.pyplot as plt

def ParticleAnalysis(acquisition,parameters,particles=None,mask=np.zeros((1))):
    """
    Perform segmentation and analysis of images of particles.
    
    Parameters
    ----------
    acquisition: Hyperpsy signal object or list of hyperspy signal objects.
        Hyperpsy signal object containing a nanoparticle image or a list of signal
         objects that contains an image at position 0 and other datasets following.
    process_param: Dictionary of parameters
        The parameters can be input manually in to a dictionary or can be generated
        using param_generator().
    particle_list: List
        List of already analysed particles that the output can be appended
        to.
    mask: Numpy array
        Numpy array of same 2D size as acquisition that contains a mask of presegmented
        particles.
        
    Returns
    -------
    list: List of Particle objects.
    """
    
    if particles==None:
        particles=Particle_list()
    
    #Check if input is list of signal objects or single one
    if isinstance(acquisition,list):
        image = acquisition[0]
        ac_types = []
        for ac in acquisition[1:]:
            if ac.metadata.Signal.signal_type == 'EDS_TEM':
                ac_types.append(ac.metadata.Signal.signal_type)
            else:
                warnings.warn("You have input data that does not have a defined signal type and therefore will not be processed."+
                              " You need to define signal_type in the metadata for anything other than the first dataset.")
    else:
        image = acquisition
        ac_types = 'Image only'
    
    if mask.sum()==0:
        labeled = process(image,parameters)
        #labels = np.unique(labeled).tolist() #some labeled number have been removed by "remove_small_holes" function
    else:
        labeled = label(mask)
        
    for region in regionprops(labeled): #'count' start with 1, 0 is background
        p = Particle()
        
        p_im = np.zeros_like(image.data)
        p_im[labeled==region.label] = image.data[labeled==region.label] - np.min(image.data[labeled==region.label])
        
        maskp = np.zeros_like(image.data)
        maskp[labeled==region.label] = 1
        
        #origin = ac_number
        #p.set_origin(origin)
        
        #Set area
        cal_area = region.area*image.axes_manager[0].scale*image.axes_manager[1].scale
        area_units = image.axes_manager[0].units+"^2"
        p.set_area(cal_area,area_units)
        
        #Set shape measures
        peri = image.axes_manager[0].scale*perimeter(maskp,neighbourhood=4)
        circularity = 4*3.14159265*p.area/(peri**2)
        p.set_circularity(circularity)
        
        #Set zoneaxis
        '''im_smooth = filters.gaussian(p_im,1)
        im_zone = np.zeros_like(im_smooth)
        im_zone[im_smooth>0] = im_smooth[im_smooth>0] - im_smooth[im_smooth>0].mean()
        im_zone[im_zone<0] = 0
        p.set_zone(zone.find_zoneaxis(im_zone))
        if p.zone != None:
            plt.imshow(im_zone)
            plt.show()'''
        
        #Set mask
        p.set_mask(maskp)
        
        if parameters.store["store_im"]==True:
            store_image(p,image)
            
        if isinstance(ac_types,list):
            for ac in acquisition:
                if ac.metadata.Signal.signal_type == 'EDS_TEM':
                    ac.set_elements(parameters.eds['elements'])
                    ac.add_lines()
                    store_spectrum(p,ac,'EDS')
                    if parameters.store["store_maps"]==True:
                        store_maps(p,ac)
                    if parameters.eds["factors"]!=False:
                        get_composition(p,parameters)
        
        particles.append(p)
        
    return(particles)
    
def store_image(particle,image):
    ii = np.where(particle.mask)
            
    box_x_min = np.min(ii[0])
    box_x_max = np.max(ii[0])
    box_y_max = np.max(ii[1])
    box_y_min = np.min(ii[1])
    pad = 5
    
    p_boxed = image.isig[(box_y_min-pad):(box_y_max+pad),(box_x_min-pad):(box_x_max+pad)]
    particle.store_im(p_boxed)
    
def store_maps(particle,ac):
    maps = ac.get_lines_intensity()
    particle.maps_gen()
    
    for map in maps:
        ii = np.where(particle.mask)
                
        box_x_min = np.min(ii[0])
        box_x_max = np.max(ii[0])
        box_y_max = np.max(ii[1])
        box_y_min = np.min(ii[1])
        pad = 5
        
        p_boxed = map.inav[(box_y_min-pad):(box_y_max+pad),(box_x_min-pad):(box_x_max+pad)]
        particle.store_map(p_boxed,p_boxed.metadata.Sample.elements[0])
        
def store_spectrum(particle,ac,stype):
    ac_particle = ac.transpose()*particle.mask
    ac_particle = ac_particle.transpose()
    ac_particle_spectrum = ac_particle.sum()
    ac_particle_spectrum.set_signal_type("EDS_TEM")
    particle.store_spectrum(ac_particle_spectrum,stype)
        
def get_composition(particle,params):
    #print(particle.spectrum['EDS'])
    bw = particle.spectrum['EDS'].estimate_background_windows(line_width=[5.0, 2.0])
    #print(bw)
    intensities = particle.spectrum['EDS'].get_lines_intensity(background_windows=bw)
    atomic_percent = particle.spectrum['EDS'].quantification(intensities, method=params.eds['method'],factors=params.eds['factors'])
    particle.store_composition(atomic_percent)
    
class parameters(object):
    """A parameters object."""
    
    def generate(self,threshold='otsu',watershed=False,invert=False,min_size=0,store_im=False,rb_kernel=0,gaussian=0,local_size=101):
        self.segment = {}
        self.segment['threshold'] = threshold
        self.segment['watershed'] = watershed
        self.segment['invert'] = invert
        self.segment['min_size'] = min_size
        self.segment['rb_kernel'] = rb_kernel
        self.segment['gaussian'] = gaussian
        self.segment['local_size'] = local_size
        
        self.store = {}
        self.store['store_im'] = store_im
        
        self.generate_eds()
        
    def generate_eds(self,eds_method=False,elements=False, factors=False, store_maps=False):
        self.eds = {}
        self.eds['method'] = eds_method
        self.eds['elements'] = elements
        self.eds['factors'] = factors
        
        self.store['store_maps'] = store_maps
    
    def save(self,filename=inspect.getfile(process).rpartition('\\')[0]+'/Parameters/parameters_current.hdf5'):
        f = h5py.File(filename,'w')
        
        segment = f.create_group("segment")
        store = f.create_group("store")
        eds = f.create_group("eds")
        
        segment.attrs["threshold"] = self.segment['threshold']
        segment.attrs["watershed"] = self.segment['watershed']
        segment.attrs["invert"] = self.segment['invert']
        segment.attrs["min_size"] = self.segment['min_size']
        segment.attrs["rb_kernel"] = self.segment['rb_kernel']
        segment.attrs["gaussian"] = self.segment['gaussian']
        store.attrs['store_im'] = self.store['store_im']
        store.attrs['store_maps'] = self.store['store_maps']
        eds.attrs['method'] = self.eds['method']
        eds.attrs['elements'] = self.eds['elements']
        eds.attrs['factors'] = self.eds['factors']
        
        f.close()
        
    def load(self,filename=inspect.getfile(process).rpartition('\\')[0]+'/Parameters/parameters_current.hdf5'):
        f = h5py.File(filename,'r')
        
        segment = f["segment"]
        store = f["store"]
        eds = f["eds"]
        
        self.segment = {}
        self.store = {}
        self.eds = {}
        
        self.segment['threshold'] = segment.attrs["threshold"]
        self.segment['watershed'] = segment.attrs["watershed"]
        self.segment['invert'] = segment.attrs["invert"]
        self.segment['min_size'] = segment.attrs["min_size"]
        self.segment['rb_kernel'] = segment.attrs["rb_kernel"]
        self.segment['gaussian'] = segment.attrs["gaussian"]
        self.store['store_im'] = store.attrs['store_im']
        self.store['store_maps'] = store.attrs['store_maps']
        self.eds['method'] = eds.attrs['method']
        self.eds['elements'] = eds.attrs['elements']
        self.eds['factors'] = eds.attrs['factors']
        
        f.close()
