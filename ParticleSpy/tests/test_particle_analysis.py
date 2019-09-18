# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:15:18 2018

@author: qzo13262
"""

import numpy.testing as nptest
import generate_test_data as gen_test
from ParticleSpy import ParticleAnalysis as PAnalysis
from ParticleSpy.ptcl_class import Particle

def test_store_image():
    mask = gen_test.generate_test_image(hspy=False)
    image = gen_test.generate_test_image(hspy=True)
    
    p = Particle()
    p.set_mask(mask)
    
    params = PAnalysis.parameters()
    params.store['pad'] = 5
    
    PAnalysis.store_image(p,image,params)
    
    nptest.assert_allclose(p.image.data,image.data[16:184,16:184])
    
def test_store_maps():
    mask = gen_test.generate_test_image(hspy=False)
    si = gen_test.generate_test_si()
    
    p = Particle()
    p.set_mask(mask)
    
    params = PAnalysis.parameters()
    params.store['pad'] = 5
    
    PAnalysis.store_maps(p,si,params)
    
    au_map = si.get_lines_intensity()[0]
    
    nptest.assert_allclose(p.maps['Au'].data,au_map.data[16:184,16:184])
    
def test_store_spectrum():
    mask = gen_test.generate_test_image(hspy=False)
    si = gen_test.generate_test_si()
    
    p = Particle()
    p.set_mask(mask)
    
    stype = 'EDS'
    
    PAnalysis.store_spectrum(p,si,stype)
    
    si_particle = si.transpose()*mask
    si_particle = si_particle.transpose()
    si_particle_spectrum = si_particle.sum()
    
    nptest.assert_allclose(p.spectrum['EDS'].data,si_particle_spectrum.data)

def test_get_composition():
    mask = gen_test.generate_test_image(hspy=False)
    si = gen_test.generate_test_si()
    
    p = Particle()
    p.set_mask(mask)
    
    stype = 'EDS'
    
    PAnalysis.store_spectrum(p,si,stype)
    
    params = PAnalysis.parameters()
    params.generate()
    params.generate_eds(eds_method='CL',elements=['Au','Pd'],factors=[1.0,1.0])
    
    PAnalysis.get_composition(p,params)
    
    nptest.assert_allclose(p.composition['Au'],46.94530019)
    
def test_particleanalysis():
    image = gen_test.generate_test_image(hspy=True)
    mask = gen_test.generate_test_image(hspy=False)
    si = gen_test.generate_test_si()
    
    ac = [image,si]
    
    params = PAnalysis.parameters()
    params.generate(store_im=True)
    params.generate_eds(eds_method='CL',elements=['Au','Pd'],factors=[1.0,1.0],store_maps=True)
    
    p_list = PAnalysis.ParticleAnalysis(ac,params,mask=mask)
    
    p = p_list.list[0]
    
    nptest.assert_almost_equal(p.properties['area']['value'],20069.0)
    assert p.properties['area']['units'] == 'nm^2'
    nptest.assert_almost_equal(p.properties['circularity']['value'],0.9095832157785668)
    assert p.zone == None
    nptest.assert_allclose(p.mask,mask)
    nptest.assert_allclose(p.image.data,image.data[16:184,16:184])
    au_map = si.get_lines_intensity()[0]
    nptest.assert_allclose(p.maps['Au'].data,au_map.data[16:184,16:184])
    si_particle = si.transpose()*mask
    si_particle = si_particle.transpose()
    si_particle_spectrum = si_particle.sum()
    nptest.assert_allclose(p.spectrum['EDS'].data,si_particle_spectrum.data)
    nptest.assert_allclose(p.composition['Au'],46.94530019)
