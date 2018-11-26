# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:48:31 2018

@author: qzo13262
"""

from ParticleAnalysis import ParticleAnalysis, parameters
from SegUI import SegUI
from ptcl_class import Particle_list

from particle_io import load_plist

def load(filename):
    p_list = load_plist(filename)
    return(p_list)