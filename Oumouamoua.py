import numpy as np
from Settings import *

objname = "1I/’Oumouamoua"
r2_vec = np.array([3.515868886595499e-2, -3.162046390773074, 4.493983111703389])   * au # Au
v2_vec = np.array([-2.317577766980901e-3, 9.843360903693031e-3, -1.541856855538041e-2])* au /dayins  # Au/day

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