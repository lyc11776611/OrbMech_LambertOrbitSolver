import numpy as np
import matplotlib.pyplot as plt
import astropy as ap

import os

from Settings import *

# from Oumouamoua import *
from Borisov import *


r0 = np.sum(r0_vec**2)**0.5
v0 = np.sum(v0_vec**2)**0.5

r2 = np.sum(r2_vec**2)**0.5
v2 = np.sum(v2_vec**2)**0.5

vr0 = np.dot(v0_vec,r0_vec)/r0
vr2 = np.dot(v2_vec,r2_vec)/r2

alpha0 = 2/r0-v0**2/mu
alpha2 = 2/r2-v2**2/mu

from orbit_propagator import *
from lambert_solver import *
from orbit_elements import *

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
def test_lambert(Dt1,Dt2,lenday=730,rdv=True):

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
        if rdv:
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
        if rdv:
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
        if rdv:
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

    test_lambert(175*dayins,500*dayins,rdv=False)

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

    test_lambert(550*dayins,1050*dayins,lenday=365*7,rdv=True)

if __name__ == '__main__':

    # orbelms = get_orbital_element(r2_vec,v2_vec)
    # print(orbelms)
    # print(reftime+orbelms[5]*dayins*np.timedelta64(1, "s"))
    # exit()

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
    test_lambert((test_depart - reftime)/ np.timedelta64(1, "s"),(test_arrive - reftime)/ np.timedelta64(1, "s"),lenday=365*orbitmaxtime,rdv=RDV)