import numpy as np
import matplotlib.pyplot as plt
import astropy as ap

import os

au = 1.496e11

dayins = 3600*24

maxiter = 1000

RDV = False

MAXDV = 20

# Earth Orbit
r0_vec = np.array([-1.796136509111975e-1, 9.667949206859814e-1, -3.668681017942158e-5]) * au  # Au
v0_vec = np.array([-1.720038360888334e-2, -3.211186197806460e-3, 7.927736735960840e-7]) * au /dayins # Au/day

reftime = np.datetime64('2017-01-01T00:00:00')

# objname = "1I/’Oumouamoua"
# r2_vec = np.array([3.515868886595499e-2, -3.162046390773074, 4.493983111703389])   * au # Au
# v2_vec = np.array([-2.317577766980901e-3, 9.843360903693031e-3, -1.541856855538041e-2])* au /dayins  # Au/day

depart1 = np.datetime64('2017-01-01T00:00:00')
depart2 = np.datetime64('2018-01-01T00:00:00')

arrive1 = np.datetime64('2017-08-01T00:00:00')
arrive2 = np.datetime64('2019-02-01T00:00:00')

# For plotting orbit in 3D
test_depart = np.datetime64('2017-07-01T00:00:00')
test_arrive = np.datetime64('2017-09-09T00:00:00')
# test_depart = np.datetime64('2017-06-15T00:00:00')
# test_arrive = np.datetime64('2018-03-01T00:00:00')
orbitmaxtime = 3 # yr


objname = "2I/Borisov"
r2_vec = np.array([7.249472033259724, 14.61063037906177, 14.24274452216359])   * au # Au
v2_vec = np.array([-8.241709369476881e-3, -1.156219024581502e-2, -1.317135977481448e-2])* au /dayins  # Au/day
#
# depart1 = np.datetime64('2017-01-01T00:00:00')
# depart2 = np.datetime64('2020-08-01T00:00:00')
#
# arrive1 = np.datetime64('2019-06-01T00:00:00')
# arrive2 = np.datetime64('2022-02-01T00:00:00')
#
# # For plotting orbit in 3D
# # test_depart = np.datetime64('2018-07-01T00:00:00')
# # test_arrive = np.datetime64('2019-11-01T00:00:00')
# test_depart = np.datetime64('2019-03-01T00:00:00')
# test_arrive = np.datetime64('2020-08-01T00:00:00')
# orbitmaxtime = 7 # yr

G = 6.67e-11
Mearth = 5.974e24
Msun = 1.989e30

acc33 = 1e-5
acc52 = 1e-5

r0 = np.sum(r0_vec**2)**0.5
v0 = np.sum(v0_vec**2)**0.5

r2 = np.sum(r2_vec**2)**0.5
v2 = np.sum(v2_vec**2)**0.5

vr0 = np.dot(v0_vec,r0_vec)/r0
vr2 = np.dot(v2_vec,r2_vec)/r2

mu = G * Msun

alpha0 = 2/r0-v0**2/mu
alpha2 = 2/r2-v2**2/mu

def S(z):

    if z > 0:
        return (z**0.5-np.sin(z**0.5))/(z**0.5)**3

    elif z==0:
        return 1/6.0

    else:
        return (np.sinh((-z)**0.5)-(-z)**0.5)/((-z)**0.5)**3

def C(z):

    if z > 0:
        return (1-np.cos(z**0.5))/z

    elif z == 0:
        return 1/2.0

    else:
        return (np.cosh((-z)**0.5)-1)/-z

# z = np.linspace(-0,500,1000)
# ss = np.zeros_like(z)
# cs = np.zeros_like(z)
#
# for i in range(len(z)):
#     ss[i] = S(z[i])
#     cs[i] = C(z[i])

def find_kai(kai0,alpha,r0,vr0,Dt,mu):

    kai = kai0
    count = 0
    relax_fac = 1
    while True:
        z = alpha*kai**2

        fkai = r0*vr0/(mu)**0.5*kai**2*C(z)+(1-alpha*r0)*kai**3*S(z)+r0*kai-mu**0.5*Dt
        fkaip = r0*vr0/(mu)**0.5*kai*(1-alpha*kai**2*S(z))+(1-alpha*r0)*kai**2*C(z)+r0

        ratio = fkai/fkaip

        if np.abs(ratio)<acc33:
            break

        if count>maxiter:
            # ratio*=relax_fac
            # count = 0
            # relax_fac*=0.9
            # print("Count reset in kai")
            raise ValueError("Cannot converge in Kai search")

        if np.isnan(ratio):
            raise ValueError("ratio is nan, no way you can solve it_1")

        kai = kai-ratio
        count += 1

    return kai

def get_fgrv(alpha,r0,Dt,kai,r0_vec,v0_vec):

    f = 1-kai**2/r0*C(alpha*kai**2)
    g = Dt-1/mu**0.5*kai**3*S(alpha*kai**2)
    r_vec = f*r0_vec+g*v0_vec
    r = np.sum(r_vec**2)**0.5
    fp = mu**0.5/r/r0*(alpha*kai**3*S(alpha*kai**2)-kai)
    gp = 1 - kai**2/r*C(alpha*kai**2)
    v_vec = fp*r0_vec + gp * v0_vec

    return [f,g,r_vec,v_vec]

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

# Function 3.3 and 3.4
def plot_orbit_test():

    ts = np.linspace(0,730*3600*24,100)
    r_vec_s1 = np.zeros((len(ts), 3))
    r_vec_s2 = np.zeros((len(ts), 3))
    for it in range(len(ts)):

        Dt = ts[it]

        Kai0 = mu ** 0.5 * np.abs(alpha0) * Dt
        kai = find_kai(Kai0,alpha0,r0,vr0,Dt,mu)
        f,g,r_vec,v_vec = get_fgrv(alpha0,r0,Dt,kai,r0_vec,v0_vec)

        r_vec_s1[it, :] = r_vec/au

        Kai0 = mu ** 0.5 * np.abs(alpha2) * Dt
        kai = find_kai(Kai0,alpha2,r2,vr2,Dt,mu)
        f,g,r_vec,v_vec = get_fgrv(alpha2,r2,Dt,kai,r2_vec,v2_vec)

        r_vec_s2[it,:] = r_vec/au

        print(it)

    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(r_vec_s1[:,0],r_vec_s1[:,1],r_vec_s1[:,2])
    ax.plot(r_vec_s2[:,0],r_vec_s2[:,1],r_vec_s2[:,2])
    ax.scatter(0,0,0,c='red',marker="*")
    plt.show()

# Function 5.2 included
def test_lambert(Dt1,Dt2,lenday=730,rev=True):

    ts = np.linspace(0,lenday*dayins,100)
    r_vec_s1 = np.zeros((len(ts), 3))
    r_vec_s2 = np.zeros((len(ts), 3))
    for it in range(len(ts)):

        Dt = ts[it]

        Kai0 = mu ** 0.5 * np.abs(alpha0) * Dt
        kai = find_kai(Kai0,alpha0,r0,vr0,Dt,mu)
        f,g,r_vec,v_vec = get_fgrv(alpha0,r0,Dt,kai,r0_vec,v0_vec)

        r_vec_s1[it, :] = r_vec/au

        Kai0 = mu ** 0.5 * np.abs(alpha2) * Dt
        kai = find_kai(Kai0,alpha2,r2,vr2,Dt,mu)
        f,g,r_vec,v_vec = get_fgrv(alpha2,r2,Dt,kai,r2_vec,v2_vec)

        r_vec_s2[it,:] = r_vec/au

        print(it)

    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(r_vec_s1[:,0],r_vec_s1[:,1],r_vec_s1[:,2],label="Earth")
    ax.plot(r_vec_s2[:,0],r_vec_s2[:,1],r_vec_s2[:,2],label="Object")
    ax.scatter(0,0,0,c='red',marker="*",label="Sun")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel("Coordinate (AU)")

    # Do a Dt and start from 0
    # Dt = 300*dayins
    # Dt1 = 100*dayins
    # Dt2 = Dt + Dt1

    Dt = Dt2-Dt1

    Kai0 = mu ** 0.5 * np.abs(alpha0) * Dt1
    kai = find_kai(Kai0, alpha0, r0, vr0, Dt1, mu)
    f, g, r_vec1, v_vec1 = get_fgrv(alpha0, r0, Dt1, kai, r0_vec, v0_vec)

    r_vec1_m = r_vec1 / au
    ax.scatter(*r_vec1_m)

    Kai0 = mu ** 0.5 * np.abs(alpha2) * Dt2
    kai = find_kai(Kai0, alpha2, r2, vr2, Dt2, mu)
    f, g, r_vec2, v_vec2 = get_fgrv(alpha2, r2, Dt2, kai, r2_vec, v2_vec)

    r_vec2_m = r_vec2 / au
    ax.scatter(*r_vec2_m)

    # Get the orbit between this two points

    try:
        DTheta,A = get_DTheta_A(r_vec1,r_vec2,True)

        tv1_vec, tv2_vec = solve_z_get_v(r_vec1,r_vec2,Dt,A)
        dv1 = np.sum((tv1_vec-v_vec1)**2)**0.5
        dv2 = np.sum((tv2_vec-v_vec2)**2)**0.5
        if rev:
            dv_prog = (dv1 + dv2)/1000.0 # to km/s
        else:
            dv_prog = (dv1) / 1000.0  # to km/s
    except:
        dv_prog = np.nan

    try:
        DTheta, A = get_DTheta_A(r_vec1, r_vec2, False)

        tv1_vec, tv2_vec = solve_z_get_v(r_vec1, r_vec2, Dt, A)
        dv1 = np.sum((tv1_vec - v_vec1) ** 2) ** 0.5
        dv2 = np.sum((tv2_vec - v_vec2) ** 2) ** 0.5
        if rev:
            dv_retr = (dv1 + dv2)/1000.0 # to km/s
        else:
            dv_retr = (dv1) / 1000.0  # to km/s
    except:
        dv_retr = np.nan

    if np.isnan(np.nanmin([dv_retr,dv_prog])):
        raise ValueError("NAN!!!")
    elif np.nanargmin([dv_retr,dv_prog]) == 1:
        DTheta,A = get_DTheta_A(r_vec1,r_vec2,True)

        tv1_vec, tv2_vec = solve_z_get_v(r_vec1,r_vec2,Dt,A)
        dv1 = np.sum((tv1_vec-v_vec1)**2)**0.5
        dv2 = np.sum((tv2_vec-v_vec2)**2)**0.5
        if rev:
            dv_prog = (dv1 + dv2)/1000.0 # to km/s
        else:
            dv_prog = (dv1) / 1000.0  # to km/s
        dv = dv_prog
    else:
        dv = dv_retr


    tr1 = np.sum(r_vec1 ** 2) ** 0.5
    tv10 = np.sum(v_vec1 ** 2) ** 0.5
    tv11 = np.sum(tv1_vec ** 2) ** 0.5

    tr2 = np.sum(r_vec2 ** 2) ** 0.5
    tv21 = np.sum(tv2_vec ** 2) ** 0.5
    tv20 = np.sum(v_vec2 ** 2) ** 0.5

    tv10r = np.dot(v_vec1, r_vec1) / tr1
    tv11r = np.dot(tv1_vec, r_vec1) / tr1
    tv21r = np.dot(tv2_vec, r_vec2) / tr2
    tv20r = np.dot(v_vec2, r_vec2) / tr2

    alphat = 2 / tr1 - tv11 ** 2 / mu

    trans_ts = np.linspace(0,Dt,100)
    r_vec_s_t = np.zeros((len(trans_ts), 3))



    for it in range(len(trans_ts)):

        Dt_temp = trans_ts[it]


        Kai0 = mu ** 0.5 * np.abs(alphat) * Dt_temp
        kai = find_kai(Kai0,alphat,tr1,tv11r,Dt_temp,mu)
        f,g,r_vec,v_vec = get_fgrv(alphat,tr1,Dt_temp,kai,r_vec1,tv1_vec)

        r_vec_s_t[it, :] = r_vec/au

    ax.plot(r_vec_s_t[:, 0], r_vec_s_t[:, 1], r_vec_s_t[:, 2],label="Transfer")
    ax.legend()
    plt.show()
    print("Dv_rendez-vous=%.2e" % (dv))
    print("Dv_rendez-vous=%.2e" % (dv))

def get_dv_rdv(Dt1,Dt2):

    # Do a Dt and start from 0
    Dt = Dt2 - Dt1

    Kai0 = mu ** 0.5 * np.abs(alpha0) * Dt1
    kai = find_kai(Kai0, alpha0, r0, vr0, Dt1, mu)
    f, g, r_vec1, v_vec1 = get_fgrv(alpha0, r0, Dt1, kai, r0_vec, v0_vec)

    Kai0 = mu ** 0.5 * np.abs(alpha2) * Dt2
    kai = find_kai(Kai0, alpha2, r2, vr2, Dt2, mu)
    f, g, r_vec2, v_vec2 = get_fgrv(alpha2, r2, Dt2, kai, r2_vec, v2_vec)

    # Get the orbit between this two points

    try:
        DTheta,A = get_DTheta_A(r_vec1,r_vec2,True)

        tv1_vec, tv2_vec = solve_z_get_v(r_vec1,r_vec2,Dt,A)

        dv1 = np.sum((tv1_vec-v_vec1)**2)**0.5
        dv2 = np.sum((tv2_vec-v_vec2)**2)**0.5
        dv_prog = (dv1 + dv2)/1000.0 # to km/s
    except:
        dv_prog = np.nan

    try:
        DTheta, A = get_DTheta_A(r_vec1, r_vec2, False)

        tv1_vec, tv2_vec = solve_z_get_v(r_vec1, r_vec2, Dt, A)

        dv1 = np.sum((tv1_vec - v_vec1) ** 2) ** 0.5
        dv2 = np.sum((tv2_vec - v_vec2) ** 2) ** 0.5
        dv_retr = (dv1 + dv2) / 1000.0  # to km/s
    except:
        dv_retr = np.nan


    return np.nanmin([dv_prog,dv_retr])

def get_dv_fby(Dt1,Dt2):

    # Do a Dt and start from 0
    Dt = Dt2 - Dt1

    Kai0 = mu ** 0.5 * np.abs(alpha0) * Dt1
    kai = find_kai(Kai0, alpha0, r0, vr0, Dt1, mu)
    f, g, r_vec1, v_vec1 = get_fgrv(alpha0, r0, Dt1, kai, r0_vec, v0_vec)

    Kai0 = mu ** 0.5 * np.abs(alpha2) * Dt2
    kai = find_kai(Kai0, alpha2, r2, vr2, Dt2, mu)
    f, g, r_vec2, v_vec2 = get_fgrv(alpha2, r2, Dt2, kai, r2_vec, v2_vec)

    # Get the orbit between this two points

    try:
        DTheta,A = get_DTheta_A(r_vec1,r_vec2,True)

        tv1_vec, tv2_vec = solve_z_get_v(r_vec1,r_vec2,Dt,A)

        dv1 = np.sum((tv1_vec-v_vec1)**2)**0.5
        dv_prog = (dv1)/1000.0 # to km/s
    except:
        dv_prog = np.nan

    try:
        DTheta, A = get_DTheta_A(r_vec1, r_vec2, False)

        tv1_vec, tv2_vec = solve_z_get_v(r_vec1, r_vec2, Dt, A)

        dv1 = np.sum((tv1_vec - v_vec1) ** 2) ** 0.5
        dv_retr = (dv1) / 1000.0  # to km/s
    except:
        dv_retr = np.nan


    return np.nanmin([dv_prog,dv_retr])

def Q3():
    Dt1_stamp0 = (np.datetime64('2017-01-01T00:00:00') - np.datetime64('2017-01-01T00:00:00'))/ np.timedelta64(1, "s")
    Dt1_stamp1 = (np.datetime64('2018-01-01T00:00:00') - np.datetime64('2017-01-01T00:00:00'))/ np.timedelta64(1, "s")

    Dt2_stamp0 = (np.datetime64('2017-08-01T00:00:00') - np.datetime64('2017-01-01T00:00:00'))/ np.timedelta64(1, "s")
    Dt2_stamp1 = (np.datetime64('2019-03-01T00:00:00') - np.datetime64('2017-01-01T00:00:00'))/ np.timedelta64(1, "s")

    # Make Porkchop plot
    Dt1s = np.linspace(Dt1_stamp0,Dt1_stamp1,60)
    Dt2s = np.linspace(Dt2_stamp0,Dt2_stamp1,30)

    grd_Dt1s,grd_Dt2s = np.meshgrid(Dt1s,Dt2s)
    grd_dv = np.zeros_like(grd_Dt1s)

    for i1 in range(len(Dt1s)):
        for i2 in range(len(Dt2s)):

            Dt1 = Dt1s[i1]
            Dt2 = Dt2s[i2]

            tmp = get_dv_fby(Dt1, Dt2)

            if tmp>20:
                tmp = np.nan

            grd_dv[i2,i1] = tmp

    plt.contourf(grd_Dt1s/dayins,grd_Dt2s/dayins,grd_dv,cmap="jet")
    plt.xlabel("Dt1 Stamp")
    plt.ylabel("Dt2 Stamp")
    plt.colorbar()
    plt.show()

    # Dt1s = np.linspace(Dt1_stamp0,Dt1_stamp1,120)
    # Dts =  np.linspace(100*dayins,300*dayins,50)
    #
    # grd_Dt1s,grd_Dts = np.meshgrid(Dt1s,Dts)
    # grd_dv = np.zeros_like(grd_Dt1s)
    #
    # for i1 in range(len(Dt1s)):
    #     for i2 in range(len(Dts)):
    #
    #         Dt1 = Dt1s[i1]
    #         Dt2 = Dts[i2] + Dt1
    #
    #         grd_dv[i2,i1] = get_dv(Dt1,Dt2)
    #
    # plt.contourf(grd_Dt1s/dayins,grd_Dts/dayins,grd_dv,cmap="jet")
    # plt.xlabel("Dt1 Stamp")
    # plt.ylabel("Dt Stamp")
    # plt.colorbar()
    # plt.show()

    test_lambert(175*dayins,500*dayins,rev=False)

def Q4():
    Dt1_stamp0 = (np.datetime64('2017-01-01T00:00:00') - np.datetime64('2017-01-01T00:00:00'))/ np.timedelta64(1, "s")
    Dt1_stamp1 = (np.datetime64('2020-08-01T00:00:00') - np.datetime64('2017-01-01T00:00:00'))/ np.timedelta64(1, "s")

    Dt2_stamp0 = (np.datetime64('2019-06-01T00:00:00') - np.datetime64('2017-01-01T00:00:00'))/ np.timedelta64(1, "s")
    Dt2_stamp1 = (np.datetime64('2022-02-01T00:00:00') - np.datetime64('2017-01-01T00:00:00'))/ np.timedelta64(1, "s")

    # Make Porkchop plot
    Dt1s = np.linspace(Dt1_stamp0,Dt1_stamp1,60)
    Dt2s = np.linspace(Dt2_stamp0,Dt2_stamp1,30)

    grd_Dt1s,grd_Dt2s = np.meshgrid(Dt1s,Dt2s)
    grd_dv = np.zeros_like(grd_Dt1s)

    for i1 in range(len(Dt1s)):
        for i2 in range(len(Dt2s)):

            Dt1 = Dt1s[i1]
            Dt2 = Dt2s[i2]

            tmp = get_dv_fby(Dt1, Dt2)

            if tmp>20:
                tmp = np.nan

            grd_dv[i2,i1] = tmp

    plt.contourf(grd_Dt1s/dayins,grd_Dt2s/dayins,grd_dv,cmap="jet")
    plt.xlabel("Dt1 Stamp")
    plt.ylabel("Dt2 Stamp")
    plt.colorbar()
    plt.show()

    # Dt1s = np.linspace(Dt1_stamp0,Dt1_stamp1,120)
    # Dts =  np.linspace(100*dayins,300*dayins,50)
    #
    # grd_Dt1s,grd_Dts = np.meshgrid(Dt1s,Dts)
    # grd_dv = np.zeros_like(grd_Dt1s)
    #
    # for i1 in range(len(Dt1s)):
    #     for i2 in range(len(Dts)):
    #
    #         Dt1 = Dt1s[i1]
    #         Dt2 = Dts[i2] + Dt1
    #
    #         grd_dv[i2,i1] = get_dv(Dt1,Dt2)
    #
    # plt.contourf(grd_Dt1s/dayins,grd_Dts/dayins,grd_dv,cmap="jet")
    # plt.xlabel("Dt1 Stamp")
    # plt.ylabel("Dt Stamp")
    # plt.colorbar()
    # plt.show()

    test_lambert(550*dayins,1050*dayins,lenday=365*7,rev=True)

if __name__ == '__main__':

    orbelms = get_orbital_element(r2_vec,v2_vec)
    print(orbelms)
    print(reftime+orbelms[5]*dayins*np.timedelta64(1, "s"))
    exit()

    Dt1_stamp0 = (depart1 - reftime)/ np.timedelta64(1, "s")
    Dt1_stamp1 = (depart2 - reftime)/ np.timedelta64(1, "s")

    Dt2_stamp0 = (arrive1 - reftime)/ np.timedelta64(1, "s")
    Dt2_stamp1 = (arrive2 - reftime)/ np.timedelta64(1, "s")


    # Make Porkchop plot
    Dt1s = np.linspace(Dt1_stamp0,Dt1_stamp1,100)
    Dt2s = np.linspace(Dt2_stamp0,Dt2_stamp1,100)

    grd_Dt1s,grd_Dt2s = np.meshgrid(Dt1s,Dt2s)
    grd_dv = np.zeros_like(grd_Dt1s)

    for i1 in range(len(Dt1s)):
        for i2 in range(len(Dt2s)):

            Dt1 = Dt1s[i1]
            Dt2 = Dt2s[i2]

            if RDV:
                tmp = get_dv_rdv(Dt1, Dt2)
            else:
                tmp = get_dv_fby(Dt1, Dt2)

            if tmp>MAXDV:
                tmp = np.nan

            grd_dv[i2,i1] = tmp

    fig, ax = plt.subplots(figsize=(13,5))

    if RDV:
        title = objname + " rendez-vous"
    else:
        title = objname + " fly-by"

    c=ax.contourf(reftime+grd_Dt1s*np.timedelta64(1, "s"),reftime+grd_Dt2s*np.timedelta64(1, "s"),grd_dv,cmap="jet")
    ax.scatter(test_depart,test_arrive,s=250,c="purple",marker="*",label="Orbit shown")
    ax.set_xlabel("Departure time")
    ax.set_ylabel("Arrival time")
    ax.grid(linestyle="dashed")
    ax.legend()
    ax.set_title(title)
    fig.colorbar(c,ax=ax,label="$\Delta V (km/s)$")
    plt.show()

    # Dt1s = np.linspace(Dt1_stamp0,Dt1_stamp1,120)
    # Dts =  np.linspace(100*dayins,300*dayins,50)
    #
    # grd_Dt1s,grd_Dts = np.meshgrid(Dt1s,Dts)
    # grd_dv = np.zeros_like(grd_Dt1s)
    #
    # for i1 in range(len(Dt1s)):
    #     for i2 in range(len(Dts)):
    #
    #         Dt1 = Dt1s[i1]
    #         Dt2 = Dts[i2] + Dt1
    #
    #         grd_dv[i2,i1] = get_dv(Dt1,Dt2)
    #
    # plt.contourf(grd_Dt1s/dayins,grd_Dts/dayins,grd_dv,cmap="jet")
    # plt.xlabel("Dt1 Stamp")
    # plt.ylabel("Dt Stamp")
    # plt.colorbar()
    # plt.show()

    # Plot orbit.
    test_lambert((test_depart - reftime)/ np.timedelta64(1, "s"),(test_arrive - reftime)/ np.timedelta64(1, "s"),lenday=365*orbitmaxtime,rev=RDV)