import random
from pathlib import Path

import hyperspy.api as hs
import numpy as np
from ParticleSpy import api as ps
from ParticleSpy.segimgs import remove_large_objects
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.naive_bayes import GaussianNB

np.random.seed(10)
random.seed(10)

def test_clustering():
    
    data = hs.load(str(Path(__file__).parent.parent / 'Data/SiO2 HAADF Image.hspy'))
    
    params = ps.parameters()
    params.generate(threshold='otsu',watershed=True,watershed_size=5,watershed_erosion=0,min_size=5,rb_kernel=100)
    particles = ps.ParticleAnalysis(data,params)
    
    new_plists = particles.cluster_particles(properties=['area','circularity'])
    assert len(new_plists[0].list) == 11



def test_clustering_all():
    
    data = hs.load(str(Path(__file__).parent.parent / 'Data/SiO2 HAADF Image.hspy'))
    param_list = open(str(Path(__file__).parent.parent / 'Data/test_parameters.dat'), 'r')

    for line in param_list:

        line = line.strip("\n")
        t_p, p_num = line.split(';')
        t_p = t_p.split(',')
        params = ps.parameters()
        params.generate(threshold=t_p[0], watershed=bool(int(t_p[1])), watershed_size=int(t_p[2]), watershed_erosion=int(t_p[3]), invert= bool(int(t_p[4])), min_size=int(t_p[5]), rb_kernel=int(t_p[6]), gaussian=int(t_p[7]), local_size=int(t_p[8]))
        particles = ps.ParticleAnalysis(data,params)
        new_plists = particles.cluster_particles(properties=['area','circularity'])
        assert len(new_plists[0].list) == int(p_num)
    
    param_list.close()

def test_learn_clustering():
    
    data = hs.load(str(Path(__file__).parent.parent / 'Data/SiO2 HAADF Image.hspy'))

    mask = ps.ClusterLearn(data, DBSCAN())

    params = ps.parameters()
    params.generate()
    particles = ps.ParticleAnalysis(data, params, mask=mask)
    new_plists = particles.cluster_particles(properties=['area'])
    assert len(new_plists[0].list) == 268

def test_train_clustering():
    
    data = hs.load(str(Path(__file__).parent.parent / 'Data/SiO2 HAADF Image.hspy'))
    maskfile = Image.open(str(Path(__file__).parent.parent / 'Data/trainingmask.png'))
    mask = np.asarray(maskfile)

    params = ps.trainableParameters()
    params.setGaussian()
    params.setDiffGaussian()
    params.setMedian()
    params.setMinimum()
    params.setMaximum()
    params.setSobel()
    params.setHessian()
    params.setLaplacian()
    params.setMembrane()
    params.setGlobalSigma(1)
    params.setGlobalDiskSize(20)
    params.setGlobalPrefilter(1)
    
    _, clf = ps.ClusterTrained(data, mask, GaussianNB(), parameters=params)
    labels = ps.ClassifierSegment(clf, data.data, parameters=params)
    labels = ps.toggle_channels(labels)
    labels = ps.toggle_channels(labels)
    labels = 2 - labels
    labels = remove_large_objects(labels)

    params = ps.parameters()
    params.generate()
    particles = ps.ParticleAnalysis(data, params, mask=labels)
    new_plists = particles.cluster_particles(properties=['area'])
    assert len(new_plists[0].list) == 9

