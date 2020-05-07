from ParticleSpy import api as ps
import hyperspy.api as hs
from pathlib import Path

def test_clustering():
    
    data = hs.load(str(Path(__file__).parent.parent / 'Data/SiO2 HAADF Image.hspy'))
    
    params = ps.parameters()
    params.generate(threshold='otsu',watershed=True,min_size=5,rb_kernel=100)
    
    particles = ps.ParticleAnalysis(data,params)
    
    new_plists = particles.cluster_particles(properties=['area','circularity'])
    
    assert len(new_plists[0].list) == 45 or len(new_plists[0].list) == 57 or len(new_plists[0].list) == 43 or len(new_plists[0].list) == 59 or len(new_plists[0].list) == 99