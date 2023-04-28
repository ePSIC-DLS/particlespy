# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:48:31 2018

@author: qzo13262
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from particlespy.particle_analysis import particle_analysis, parameters, particle_analysis_series, time_series_analysis
from particlespy.segimgs import *
from particlespy.seg_ui import seg_ui
from particlespy.ptcl_class import particle_list, particle

from particlespy.particle_load import load_plist
from particlespy.radial_profile import radial_profile, plot_profile

def load(filename):
    p_list = load_plist(filename)
    return(p_list)

def plot(particle_lists,prop_list=['area'],**kwargs):
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
        if isinstance(prop_list,str):
            fig = plt.figure()
            ax = fig.add_subplot(111)
        elif len(prop_list) == 1 or len(prop_list) == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        elif len(prop_list) == 3:
            fig3d = plt.figure()
            ax = fig3d.add_subplot(111, projection='3d')
        
        for p in particle_lists:
            if isinstance(prop_list,str):
                p._plot_one_property(prop_list,ax,**kwargs)
            else:
                if len(prop_list) == 1:
                    p._plot_one_property(prop_list[0],ax,**kwargs)
                elif len(prop_list) == 2:
                    p._plot_two_properties(prop_list,ax,**kwargs)
                elif len(prop_list) == 3:
                    p._plot_three_properties(prop_list,ax,**kwargs)
                else:
                    print("Can only plot one or two properties, please change the length of the property list.")
                    break
        
        plt.show()

def test():
    print('Test')