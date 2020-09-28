from ParticleSpy import api as ps
import hyperspy.api as hs
from pathlib import Path

def test_clustering():
    
    data = hs.load(str(Path(__file__).parent.parent / 'Data/SiO2 HAADF Image.hspy'))
    
    params = ps.parameters()
    params.generate(threshold='otsu',watershed=True,watershed_size=5,watershed_erosion=0,min_size=5,rb_kernel=100)
    particles = ps.ParticleAnalysis(data,params)
    
    new_plists = particles.cluster_particles(properties=['area','circularity'])
    
    assert len(new_plists[0].list) == 3 or len(new_plists[0].list) == 45 or len(new_plists[0].list) == 57 or len(new_plists[0].list) == 43 or len(new_plists[0].list) == 59 or len(new_plists[0].list) == 99

def test_clustering_all():
    
    data = hs.load(str(Path(__file__).parent.parent / 'Data/SiO2 HAADF Image.hspy'))
    param_list = open(str(Path(__file__).parent.parent / 'Data/test_parameters.dat'), 'r')

    for line in param_list:

        line = line.strip("\n")
        test_params, test_results = line.split(' ; ')
        t_p = test_params.split(',')
        test_results = test_results.split(',')
        params = ps.parameters()
        params.generate(threshold=t_p[0], watershed=bool(int(t_p[1])), watershed_size=int(t_p[2]), watershed_erosion=int(t_p[3]), invert= bool(int(t_p[4])), min_size=int(t_p[5]), rb_kernel=int(t_p[6]), gaussian=int(t_p[7]), local_size=int(t_p[8]))
        particles = ps.ParticleAnalysis(data,params)
        new_plists = particles.cluster_particles(properties=['area','circularity'])

        verif = False
        for value in test_results:
            print(t_p[0],value,len(new_plists[0].list))
            if len(new_plists[0].list) == int(value):
                verif = True
                break
        assert verif == True
    
    param_list.close()
test_clustering_all()