import numpy as np
from Settings import *

def get_orbital_element(r_vec,v_vec):

    r = np.sum(r_vec**2)**0.5
    v = np.sum(v_vec**2)**0.5

    vr = np.dot(r_vec,v_vec)/r

    h_vec = np.cross(r_vec,v_vec)
    h = np.sum(h_vec**2)**0.5

    i = np.arccos(h_vec[2]/h)

    N_vec = np.cross(np.array([0,0,1]),h_vec)
    N = np.sum(N_vec**2)**0.5

    if N_vec[1] >=0:
        Omega = np.arccos(N_vec[0]/N)
    else:
        Omega = 2*np.pi - np.arccos(N_vec[0]/N)

    e_vec = 1/mu*((v**2-mu/r)*r_vec-r*vr*v_vec)

    e = np.sum(e_vec**2)**0.5

    omega = np.arccos(np.dot(N_vec,e_vec)/N/e)
    if e_vec[2] <0:
        omega = 2*np.pi - omega

    theta = np.arccos(np.dot(e_vec,r_vec)/e/r)
    if vr<0:
        theta = 2*np.pi - theta

    if e<1: # Elliptical orbits
        a = h**2/mu/(1-e**2)
        return(a / au,
               i / np.pi*180.0,
               Omega/np.pi*180.0,
               e,
               omega/np.pi*180.0,
               theta/np.pi*180.0)
    if e>=1: # Hyperbolic trajectory
        q = h ** 2 / mu / (1 + e)

        # Get time of periapsis passage
        # all_thetas = np.linspace(0,theta,1000)
        # dtheta = all_thetas[1] - all_thetas[0]
        # t = np.sum(dtheta/(1+e*np.cos(all_thetas))**2)/mu**2*h**3

        t = 1/(e**2-1)*(e*np.sin(theta)/(1+e*np.cos(theta)))-1/(e**2-1)**(3/2)*np.log(((e+1)**0.5+(e-1)**0.5*np.tan(theta/2))/((e+1)**0.5-(e-1)**0.5*np.tan(theta/2)))
        t = -t/mu**2*h**3
        tp = reftime+t*np.timedelta64(1, "s")


        return (q / au,
                i / np.pi*180.0,
                Omega / np.pi * 180.0,
                e,
                omega / np.pi * 180.0,
                t/dayins)
