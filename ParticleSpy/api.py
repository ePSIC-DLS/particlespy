# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:48:31 2018

@author: qzo13262
"""

from ParticleSpy.ParticleAnalysis import ParticleAnalysis, parameters
from ParticleSpy.SegUI import SegUI
from ParticleSpy.ptcl_class import Particle_list, Particle

from ParticleSpy.particle_load import load_plist

def load(filename):
    p_list = load_plist(filename)
    return(p_list)
