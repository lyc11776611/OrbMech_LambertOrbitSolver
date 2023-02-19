import numpy as np

au = 1.496e11

dayins = 3600*24

maxiter = 1000

RDV = False # Fly be or rendezvous?

MAXDV = 20

# Earth Orbit
r0_vec = np.array([-1.796136509111975e-1, 9.667949206859814e-1, -3.668681017942158e-5]) * au  # Au
v0_vec = np.array([-1.720038360888334e-2, -3.211186197806460e-3, 7.927736735960840e-7]) * au /dayins # Au/day

reftime = np.datetime64('2017-01-01T00:00:00')

G = 6.67e-11
Mearth = 5.974e24
Msun = 1.989e30

acc33 = 1e-5
acc52 = 1e-5

mu = G * Msun
