import numpy as np
from Settings import *

objname = "2I/Borisov"
r2_vec = np.array([7.249472033259724, 14.61063037906177, 14.24274452216359])   * au # Au
v2_vec = np.array([-8.241709369476881e-3, -1.156219024581502e-2, -1.317135977481448e-2])* au /dayins  # Au/day
#
depart1 = np.datetime64('2017-01-01T00:00:00')
depart2 = np.datetime64('2020-08-01T00:00:00')

arrive1 = np.datetime64('2019-06-01T00:00:00')
arrive2 = np.datetime64('2022-02-01T00:00:00')

# For plotting orbit in 3D
# test_depart = np.datetime64('2018-07-01T00:00:00')
# test_arrive = np.datetime64('2019-11-01T00:00:00')
test_depart = np.datetime64('2019-03-01T00:00:00')
test_arrive = np.datetime64('2020-08-01T00:00:00')
orbitmaxtime = 7 # yr