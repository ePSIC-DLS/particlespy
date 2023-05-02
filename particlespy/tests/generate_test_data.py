# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:03:57 2018

@author: qzo13262
"""

import numpy as np
from skimage import draw
import hyperspy.api as hs

def generate_test_image(hspy=True):
    arr = np.zeros((200, 200))
    rr, cc = draw.disk((100, 100), radius=80, shape=arr.shape)
    arr[rr, cc] = 1
    if hspy==True:
        arr = hs.signals.Signal2D(arr)
        arr.axes_manager[0].scale = 1
        arr.axes_manager[0].units = "nm"
        arr.axes_manager[1].scale = 1
        arr.axes_manager[1].units = "nm"
    return(arr)
    
def generate_test_image2(hspy=True):
    arr = np.zeros((200, 200))
    rr, cc = draw.disk((50, 50), radius=20, shape=arr.shape)
    arr[rr, cc] = 1
    rr, cc = draw.disk((150, 150), radius=30, shape=arr.shape)
    arr[rr, cc] = 1
    if hspy==True:
        arr = hs.signals.Signal2D(arr)
        arr.axes_manager[0].scale = 1
        arr.axes_manager[0].units = "nm"
        arr.axes_manager[1].scale = 1
        arr.axes_manager[1].units = "nm"
    return(arr)

def generate_test_eds():
    arr = np.zeros((200,200,2048))
    rr, cc = draw.disk((100, 100), radius=80, shape=arr.shape)
    
    x_values = np.linspace(0,2048,2048)
    y = np.zeros_like(x_values)
    for mu, sig in [(0,5),(280,7),(970,10)]:
        y = y + gaussian(x_values,mu,sig)
    
    arr[rr, cc, :] = y
    
    s = hs.signals.Signal1D(arr)
    
    s.axes_manager[2].scale = 0.01
    s.axes_manager[2].units = "keV"
    s.axes_manager[2].offset = 0
    
    s.set_signal_type('EDS_TEM')
    
    s.set_microscope_parameters(beam_energy=200)
    s.set_elements(['Au','Pd'])
    s.set_lines(['Au_La','Pd_La'])
    
    return(s)
    
def generate_test_si(signal_type='Arbitrary'):
    arr = np.zeros((200,200,2048))
    rr, cc = draw.disk((100, 100), radius=80, shape=arr.shape)
    
    x_values = np.linspace(0,2048,2048)
    y = np.zeros_like(x_values)
    for mu, sig in [(0,5),(280,7),(970,10)]:
        y = y + gaussian(x_values,mu,sig)
    
    arr[rr, cc, :] = y
    
    s = hs.signals.Signal1D(arr)
    
    s.axes_manager[2].scale = 0.01
    s.axes_manager[2].units = "keV"
    s.axes_manager[2].offset = 0
    
    s.set_signal_type(signal_type)
    
    return(s)
    
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))