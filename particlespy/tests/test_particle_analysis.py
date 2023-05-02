# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:15:18 2018

@author: qzo13262
"""

import numpy.testing as nptest
import particlespy.tests.generate_test_data as gen_test
from particlespy import particle_analysis
from particlespy.ptcl_class import particle

def test_store_image():
    mask = gen_test.generate_test_image(hspy=False)
    image = gen_test.generate_test_image(hspy=True)
    
    p = particle()
    p.set_mask(mask)
    
    params = particle_analysis.parameters()
    params.generate()
    
    particle_analysis.store_image(p,image,params)
    
    nptest.assert_allclose(p.image.data,image.data[16:184,16:184])
    
def test_store_maps():
    mask = gen_test.generate_test_image(hspy=False)
    si = gen_test.generate_test_eds()
    
    p = particle()
    p.set_mask(mask)
    
    params = particle_analysis.parameters()
    params.generate()
    
    particle_analysis.store_maps(p,si,params)
    
    au_map = si.get_lines_intensity()[0]
    
    nptest.assert_allclose(p.maps['Au'].data,au_map.data[16:184,16:184])
    
def test_store_spectrum():
    mask = gen_test.generate_test_image(hspy=False)
    si = gen_test.generate_test_eds()
    
    p = particle()
    p.set_mask(mask)
    
    stype = 'EDS_TEM'
    
    p.spectrum = {}
    particle_analysis.store_spectrum(p,si,stype)
    
    si_particle = si.transpose()*mask
    si_particle = si_particle.transpose()
    si_particle_spectrum = si_particle.sum()
    
    nptest.assert_allclose(p.spectrum['EDS_TEM'].data,si_particle_spectrum.data)

def test_get_composition():
    mask = gen_test.generate_test_image(hspy=False)
    eds = gen_test.generate_test_eds()
    
    p = particle()
    p.set_mask(mask)
    
    stype = 'EDS_TEM'
    
    p.spectrum = {}
    particle_analysis.store_spectrum(p,eds,stype)
    
    params = particle_analysis.parameters()
    params.generate()
    params.generate_eds(eds_method='CL',elements=['Au','Pd'],factors=[1.0,1.0])
    
    particle_analysis.get_composition(p,params)
    
    nptest.assert_allclose(p.composition['Au'],46.94530019)
    
def test_particleanalysis():
    image = gen_test.generate_test_image(hspy=True)
    mask = gen_test.generate_test_image(hspy=False)
    eds = gen_test.generate_test_eds()
    si = gen_test.generate_test_si()
    eels = gen_test.generate_test_si('EELS')
    
    ac = [image,eds,si,eels]
    
    params = particle_analysis.parameters()
    params.generate(store_im=True)
    params.generate_eds(eds_method='CL',elements=['Au','Pd'],factors=[1.0,1.0],store_maps=True)
    
    p_list = particle_analysis.particle_analysis(ac,params,mask=mask)
    
    p = p_list.list[0]
    
    nptest.assert_almost_equal(p.properties['area']['value'],20069.0)
    assert p.properties['area']['units'] == 'nm^2'
    nptest.assert_almost_equal(p.properties['circularity']['value'],0.9095832157785668)
    nptest.assert_almost_equal(p.properties['equivalent circular diameter']['value'],159.8519453221949)
    assert p.properties['equivalent circular diameter']['units'] == 'nm'
    #assert p.zone == None
    nptest.assert_allclose(p.mask,mask)
    nptest.assert_allclose(p.image.data,image.data[16:184,16:184])
    au_map = eds.get_lines_intensity()[0]
    nptest.assert_allclose(p.maps['Au'].data,au_map.data[16:184,16:184])
    eds_particle = eds.transpose()*mask
    eds_particle = eds_particle.transpose()
    eds_particle_spectrum = eds_particle.sum()
    nptest.assert_allclose(p.spectrum['EDS_TEM'].data,eds_particle_spectrum.data)
    nptest.assert_allclose(p.composition['Au'],46.94530019)
    nptest.assert_allclose(p.properties['x']['value'], 100.)
    nptest.assert_allclose(p.properties['y']['value'], 100.)
    nptest.assert_allclose(p.properties['bbox_area']['value'], 25281.0)
    nptest.assert_allclose(p.properties['bbox_length']['value'], 224.8599564173221)
    assert p.properties['bbox_area']['units'] == 'nm^2'
    assert p.properties['bbox_length']['units'] == 'nm'
    assert p.bbox == (21, 21, 180, 180)
    
def test_series():
    image = gen_test.generate_test_image(hspy=True)
    image2 = gen_test.generate_test_image(hspy=True)
    images = [image,image2]
    mask = gen_test.generate_test_image(hspy=False)
    mask2 = gen_test.generate_test_image(hspy=False)
    masks = [mask,mask2]
    
    params = particle_analysis.parameters()
    params.generate(store_im=True)
    
    p_list = particle_analysis.particle_analysis_series(images,params)
    
    p = p_list.list[0]
    
    nptest.assert_almost_equal(p.properties['area']['value'],20069.0)
    assert p.properties['area']['units'] == 'nm^2'
    nptest.assert_almost_equal(p.properties['circularity']['value'],0.9095832157785668)
    nptest.assert_allclose(p.mask,mask)
    nptest.assert_allclose(p.image.data,image.data[16:184,16:184])
    nptest.assert_allclose(p.properties['x']['value'], 100.)
    nptest.assert_allclose(p.properties['y']['value'], 100.)
    assert p.properties['frame']['value'] == 0
    
def test_time_series():
    image = gen_test.generate_test_image(hspy=True)
    image2 = gen_test.generate_test_image(hspy=True)
    images = [image,image2]
    mask = gen_test.generate_test_image(hspy=False)
    mask2 = gen_test.generate_test_image(hspy=False)
    masks = [mask,mask2]
    
    params = particle_analysis.parameters()
    params.generate(store_im=True)
    
    p_list = particle_analysis.particle_analysis_series(images,params)
    
    t = particle_analysis.time_series_analysis(p_list)
    
    nptest.assert_almost_equal(t['area'][:1][0],20069.0)
    
def test_normalize_boxing():
    mask = gen_test.generate_test_image2(hspy=False)
    image = gen_test.generate_test_image2(hspy=True)
    
    params = particle_analysis.parameters()
    params.generate(store_im=True)
    
    particles = particle_analysis.particle_analysis(image,params,mask=mask)
    
    particles.normalize_boxing()
    
    assert particles.list[0].image.data.shape == (68,68)

def test_params_in_out():
    params = particle_analysis.parameters()
    params.load()
    params.save()