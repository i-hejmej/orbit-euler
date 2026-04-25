import numpy as np
from numba import njit 

G = 6.6741 * (10 ** (-11))
AU = 149597870700

@njit
def _calculate_orbit_jit(M_central, dt, steps, x, y, v_x, v_y):

    Xs = np.zeros(steps)
    Ys = np.zeros(steps)
   
    for i in range(steps):
       
        Xs[i] = x
        Ys[i] = y
        
        r = np.sqrt(x**2 + y**2)
        
        dx = v_x * dt
        dy = v_y * dt
        dv_x = -G * M_central / r**3 * x * dt
        dv_y = -G * M_central / r**3 * y * dt
        
        x += dx
        y += dy
        v_x += dv_x
        v_y += dv_y
        
    return Xs, Ys

Dt = 60 * 15
T_max = float(3 * 75 * 365 * 24 * 60 * 60)
X_0 = 0
Y_0 = 0.586 * AU
V_0X = 54600
V_0Y = 0   
M_S = 1.989*10**(30)

def calculate_orbit(*, M_central = M_S, dt=Dt, time_max=T_max, x_0=X_0, y_0=Y_0, v_0x=V_0X, v_0y=V_0Y):
    '''
    Simulates a bodies' 2D orbit around the central mass. 

    Function uses Euler's method to calculate the trajectory step by step.
    Output is automaticly scaled to astronomical units. Default settings' output is the trajcetory of Halley's comet around the Sun.
    
    Parameters
    ----------
    M_central : float, optional
        The mass of the central body in kilograms. Solar mass by default (1.989e30 kg).
    dt : float, optional
        Time step used in the simulation in seconds. 900 by default (15 minutes).
    time_max : float, optional
        Time of the whole simulation in seconds. 225 years by default.
    x_0 : float, optional
        Initial X-axis position in meters. 0 by default.
    y_0 : float, optional
        Initial Y-axis position in meters. 0.586 AU by default.
    v0_x : float, optional
        Initial X-axis velocity in m/s. 54600 by default.
    v0_y : float, optional
        Initial Y-axis velocity in m/s. 0 by default.
    
    Returns
    -------
    Xs : ndarray
        One-dimensional array containing all X posisionts (in AU) of the orbiting body.
    Ys : ndarray
        One-dimensional array containing all Y posisionts (in AU) of the orbiting body.
    '''
    steps = int(time_max / dt) 
    Xs, Ys = _calculate_orbit_jit(M_central, dt, steps, x_0, y_0, v_0x, v_0y)
    Xs /= AU
    Ys /= AU
    
    return Xs, Ys
