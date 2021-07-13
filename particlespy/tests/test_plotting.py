from particlespy import api as ps
import os

def test_plotting():
    my_path = os.path.dirname(__file__)
    
    p_list = ps.load(os.path.join(my_path, 'test_particle.hdf5'))
    
    p_list.plot('area')
    p_list.plot(['area'])
    p_list.plot(['area','circularity'])
    p_list.plot(['area','circularity','area'])
    p_list.plot(['area','circularity','area','area'])
    
    ps.plot([p_list],'area')
    ps.plot([p_list],['area'])
    ps.plot([p_list],['area','circularity'])
    ps.plot([p_list],['area','circularity','area','area'])