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
    
    if mask == 'UI':
        labeled = label(np.load(inspect.getfile(process).rpartition('\\')[0]+'/Parameters/manual_mask.npy'))
    elif mask.sum()==0:
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
        
        #Set diam measures
        cal_circdiam = 2*(cal_area**0.5)/np.pi
        diam_units = image.axes_manager[0].units
        p.set_circdiam(cal_circdiam,diam_units)
        
        cal_axes_lengths = (region.major_axis_length*image.axes_manager[0].scale,region.minor_axis_length*image.axes_manager[0].scale)
        #Note: the above only works for square pixels
        p.set_axes_lengths(cal_axes_lengths,diam_units)
        
        #Set shape measures
        peri = image.axes_manager[0].scale*perimeter(maskp,neighbourhood=4)
        circularity = 4*3.14159265*p.properties['area']['value']/(peri**2)
        p.set_circularity(circularity)
        eccentricity = region.eccentricity
        p.set_eccentricity(eccentricity)
        
        #Set total image intensity
        intensity = (image.data*maskp).sum()
        p.set_intensity(intensity)
        
        #Set zoneaxis
        im_smooth = filters.gaussian(np.uint16(p_im),1)
        im_zone = np.zeros_like(im_smooth)
        im_zone[im_smooth>0] = im_smooth[im_smooth>0] - im_smooth[im_smooth>0].mean()
        im_zone[im_zone<0] = 0
        p.set_zone(zone.find_zoneaxis(im_zone))
        if p.zone != None:
            plt.imshow(im_zone)
            plt.show()
        
        #Set mask
        p.set_mask(maskp)
        
        if parameters.store["store_im"]==True:
            store_image(p,image,parameters)
            
        if isinstance(acquisition,list):
            p.spectrum = {}
            for ac in acquisition:
                if ac.metadata.Signal.signal_type == 'EDS_TEM':
                    ac.set_elements(parameters.eds['elements'])
                    ac.add_lines()
                    store_spectrum(p,ac,'EDS_TEM')
                    if parameters.store["store_maps"]:
                        store_maps(p,ac,parameters)
                    if parameters.eds["factors"]!=False:
                        get_composition(p,parameters)
                elif ac.metadata.Signal.signal_type == 'EELS':
                    ac.add_elements(parameters.eds['elements'])
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
    
def store_image(particle,image,params):
    ii = np.where(particle.mask)
            
    box_x_min = np.min(ii[0])
    box_x_max = np.max(ii[0])
    box_y_max = np.max(ii[1])
    box_y_min = np.min(ii[1])
    pad = params.store['pad']
    
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
    
    def generate(self,threshold='otsu',watershed=False,invert=False,min_size=0,store_im=False,pad=5,rb_kernel=0,gaussian=0,local_size=101):
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
        self.store['pad'] = pad
        
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
        store.attrs['pad'] = self.store['pad']
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
        self.store['pad'] = store.attrs['pad']
        self.store['store_maps'] = store.attrs['store_maps']
        self.eds['method'] = eds.attrs['method']
        self.eds['elements'] = eds.attrs['elements']
        self.eds['factors'] = eds.attrs['factors']
        
        f.close()
