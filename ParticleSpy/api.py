# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:48:31 2018

@author: qzo13262
"""

import matplotlib.pyplot as plt

from ParticleSpy.ParticleAnalysis import ParticleAnalysis, parameters, ParticleAnalysisSeries, timeseriesanalysis
from ParticleSpy.SegUI import SegUI
from ParticleSpy.ptcl_class import Particle_list, Particle

from ParticleSpy.particle_load import load_plist
from ParticleSpy.radial_profile import radial_profile, plot_profile

def load(filename):
    p_list = load_plist(filename)
    return(p_list)

def plot(particle_lists,prop_list=['area'],bins=20):
        """
        Plots properties of all particles in the Particle_lists.
        
        If one property given, displays a histogram of the chosen particle property.
        
        If two properties given, displays a scatter plot of the two properties.
        
        Parameters
        ----------
        particle_lists : list
            A list of Particle_list objects.
        prop_list : str or list
            The name of a property or a list of the properties to plot.
        bins : int
            The number of bins in the histogram if plotting one property.
            
        Examples
        --------
        
        plot([particles],['area'])
        
        """
        
        for p in particle_lists:
            if isinstance(prop_list,str):
                p._plot_one_property(prop_list,bins)
            else:
                if len(prop_list) == 1:
                    p._plot_one_property(prop_list[0],bins)
                elif len(prop_list) == 2:
                    p._plot_two_properties(prop_list)
                elif len(prop_list) == 3:
                    p._plot_three_properties(prop_list)
                else:
                    print("Can only plot one or two properties, please change the length of the property list.")
                    break
        
        plt.show()
