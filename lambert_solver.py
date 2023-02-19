from orbit_propagator import C,S
from Settings import *
import numpy as np



def get_DTheta_A(r1_vec,r2_vec,prograde=True):

    up = np.cross(r1_vec,r2_vec)[2]>=0
    r1 = np.sum(r1_vec**2)**0.5
    r2 = np.sum(r2_vec**2)**0.5

    if prograde and up:
        DTheta = np.arccos(np.dot(r1_vec,r2_vec)/r1/r2)
    elif prograde and not up:
        DTheta = 2*np.pi - np.arccos(np.dot(r1_vec, r2_vec) / r1 / r2)
    elif not prograde and not up:
        DTheta = np.arccos(np.dot(r1_vec,r2_vec)/r1/r2)
    elif not prograde and up:
        DTheta = 2*np.pi - np.arccos(np.dot(r1_vec, r2_vec) / r1 / r2)

    A = np.sin(DTheta)*(r1*r2/(1-np.cos(DTheta)))**0.5

    return [DTheta,A]

def solve_z_get_v(r1_vec,r2_vec,Dt,A,z0=0):

    r1 = np.sum(r1_vec ** 2) ** 0.5
    r2 = np.sum(r2_vec ** 2) ** 0.5

    z = z0

    # get z
    count = 0
    relax_fac = 1
    while True:
        CC = C(z)
        SS = S(z)

        y = r1+r2+A*(z*SS-1)/CC**0.5



        F = (y/CC)**1.5*SS+A*y**0.5-mu**0.5*Dt

        if z == 0:
            Fp = 2**0.5/40.0*y**1.5+A/8.0*(y**0.5+A*(1/2.0/y))**0.5
        else:
            Fp = (y/CC)**1.5*(1/2.0/z*(CC-3/2.0*SS/CC)+3/4.0*SS**2/CC)+A/8.0*(3*SS/CC*y**0.5+A*(CC/y)**0.5)

        ratio = F / Fp

        if np.isnan(ratio):
            raise ValueError("ratio is nan, no way you can solve it_1")

        if np.abs(ratio) < acc52:
            break

        if count > maxiter:
            # ratio *= relax_fac
            # count = 0
            #
            # if ratio >1e5:
            #     # Somethings wrong
             raise ValueError("Z search cannot converge")

            # relax_fac*=0.99
            #
            # if relax_fac<=0.1:
            #     break
            #
            # print("Count reset in finding z,ratio: %.2e"%z)

        z = z - ratio
        count += 1


    CC = C(z)
    SS = S(z)
    y = r1 + r2 + A * (z * SS - 1) / CC ** 0.5

    f = 1-y/r1
    g = A*(y/mu)**0.5
    fp = mu**0.5/r1/r2*(y/CC)**0.5*(z*SS-1)
    gp = 1 - y/r2

    v1_vec = 1/g*(r2_vec-f*r1_vec)
    v2_vec = 1/g*(gp*r2_vec-r1_vec)

    return (v1_vec,v2_vec)
