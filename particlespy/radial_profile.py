# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:58:37 2019

@author: qzo13262
"""
import numpy as np
import matplotlib.pyplot as plt

def radial_profile(particle,signals,plot=True, mark_radius=False):
    """
    Function to calculate and plot a radial profile of a signal from an individual 
    particle.
    
    Parameters
    ----------
    particle : ParticleSpy particle object
        The particle object.
    signals : list
        List of signals to plot, either 'Image' or element name, e.g. 'Ag'.
    plot : bool
        True if plotting, False otherwise.
    mark_radius : bool
        If true, mark the particle radius on the profile plot.
    """
    
    dist_count_dic = {}
    for sig in signals:
        if sig == 'Image':
            dist_count_dic['Image'] = concentric_scan_absolutedis(particle.image)
        else:
            dist_count_dic[sig] = concentric_scan_absolutedis(particle.maps[sig])
    
    scale = particle.image.axes_manager[0].scale
    units = particle.image.axes_manager[0].units
    
    if plot==True:
        plot_profile(dist_count_dic,scale,units, mark_radius=mark_radius, radius=particle.properties['equivalent circular diameter']['value']/2)
    return(dist_count_dic)
    
'''def sum_profiles(profiles):
    sum_profile = {}
    
    for profile in profiles:
        for sig in dist_count_dic:
            tuple(map(sum, zip(a, b)))'''

def concentric_scan_absolutedis(element_map):
    '''
    Distance unit is pixel and length is absolute length from particle center
    ----
    Return:
        dis_count_dict: keys ordered dict {dis0: countX0, dis1: countX1, ...}
    '''
    dis_count_ls=[]
    ax0_size = np.shape(element_map)[0]
    ax1_size = np.shape(element_map)[1]
    
    ax0_centre = ax0_size/2
    ax1_centre = ax1_size/2
    
    dis_count_ls = []
    for i_ax0 in range(ax0_size):
        for i_ax1 in range(ax1_size):
            dis_from_centre = int(np.sqrt((i_ax0 - ax0_centre)**2 + \
                                          (i_ax1 - ax1_centre)**2))
            dis_count_ls.append([dis_from_centre, element_map.data[i_ax0, i_ax1]])

    #sort by the first item        
    def getkey(item):
        return item[0]
    dis_count_ls = sorted(dis_count_ls, key=getkey) 
    distance_ls, count_ls = np.transpose(dis_count_ls)
    
    count_number = 0
    distance_unique = []
    count_unique = []
    count = 0
    for index in range(len(distance_ls)):
        if index < len(distance_ls)-1:
            if int(distance_ls[index]) == int(distance_ls[index+1]):
                count += count_ls[index]
                count_number += 1
            elif int(distance_ls[index]) != int(distance_ls[index+1]):
                count += count_ls[index]
                count_number += 1
                distance_unique.append(int(distance_ls[index]))
                count_unique.append(count/count_number)
                count = 0
                count_number = 0
        else:
            if int(distance_ls[index]) == int(distance_ls[index-1]):
                count += count_ls[index]
                count_number += 1
                distance_unique.append(int(distance_ls[index]))
                count_unique.append(count/count_number)
            elif int(distance_ls[index]) != int(distance_ls[index-1]):
                distance_unique.append(int(distance_ls[index]))
                count_unique.append(count_ls[index])
    
    return distance_unique, count_unique

def plot_profile(dist_count_dic, scale, units, mark_radius=False, radius=1.0, save=False, dir_save=None):
    """
    Function to plot a radial profile of particle signals.
    
    Parameters
    ----------
    dist_count_dic : dict
        Dictionary containing the distances and counts of the profile.
    scale : float
    units : str
    mark_radius : bool
        If true, mark the particle radius on the profile plot.
    radius : float
    save : bool
    dir_save : str
        Default : None
    """
        
    plt.xlabel('Distance from particle centre ('+units+')')
    plt.ylabel('Normalised intensity (counts)')
    
    for sig, count_and_dist in dist_count_dic.items():
        if sig=='Image':
            sig = 'ADF'
        plt.plot(np.array(count_and_dist[0])*scale, count_and_dist[1],label=sig)
        if mark_radius==True:
            plt.axvline(x=radius,color='k',linestyle='--')
    plt.legend()
    
    if save==True:
        plt.savefig(dir_save,bbox_inches='tight',dpi=600)