from particlespy import api as ps
import numpy.testing as nptest
import particlespy.tests.generate_test_data as gen_test
import os

def test_load_particles():
    my_path = os.path.dirname(__file__)
    
    p_list = ps.load(os.path.join(my_path, 'test_particle.hdf5'))
    p = p_list.list[0]
    
    image = gen_test.generate_test_image(hspy=True)
    mask = gen_test.generate_test_image(hspy=False)
    
    nptest.assert_almost_equal(p.properties['area']['value'],20069.0)
    assert p.properties['area']['units'] == 'nm^2'
    nptest.assert_almost_equal(p.properties['circularity']['value'],0.9095832157785668)
    nptest.assert_allclose(p.mask,mask)
    nptest.assert_allclose(p.image.data,image.data[16:184,16:184])
