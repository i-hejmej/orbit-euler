import numpy as np
from pyorbits.orbit_euler import calculate_orbit 

def test_output_array_lengths():
    dt_test = 1000
    time_max_test = 5000
    expected_length = int(time_max_test / dt_test)
    
    xs, ys = calculate_orbit(dt=dt_test, time_max=time_max_test)
    
    assert len(xs) == expected_length
    assert len(ys) == expected_length

def test_initial_conditions_scaling():
    AU = 149597870700
    x0 = 0
    y0 = 1.5 * AU  
  
    xs, ys = calculate_orbit(x_0=x0, y_0=y0, dt=10, time_max=10)
    
    assert np.isclose(xs[0], 0.0)
    assert np.isclose(ys[0], 1.5)

def test_free_fall_physics():
    x0 = 149597870700 
    y0 = 0
    
    xs, ys = calculate_orbit(x_0=x0, y_0=y0, v_0x=0, v_0y=0, dt=100, time_max=1000)
    
    for y in ys:
        assert np.isclose(y, 0.0)
    
    assert xs[1] < xs[0]