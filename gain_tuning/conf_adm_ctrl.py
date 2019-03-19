# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Configuration for tuning the gains for Admittance Control

@author: adelprete
"""

#!/usr/bin/env python
from __future__ import print_function
from numpy import matlib
from conf_common import *

#TESTS_DIR_NAME          = 'adm_ctrl_w_dx_0_d2x_0_d3x_0/'
GAINS_FILE_NAME         = 'gains_adm_ctrl'
controllers             = ['adm_ctrl']
ctrl_long_name          = 'Adm-Ctrl'

ny          = 3     # size of configuration vector (i.e. momentum)
nf          = 4     # size of the force vector

# state used to compute expected trajectory to compare to real traj (fixed_x0)
x0          = matlib.zeros((3*nf+2*ny,1))
x0[ny,0]    = 47.2 * .1     # initial CoM velocity in Y direction times robot mass

# (deprecated) state used for gain tuning, which is equal to initial simulation state
#x0_com      = 1e-3*np.matrix([[ 0.0, 0.0, 1.1e+02, -7.5e-02, 6.0e+01, -3.0e+02, -2.8e-11, 2.9e-11, 0.0, 0.0]]).T #2.6e+08, 6.7e+07]]).T

max_iter    = 1        # max number of iterations of algorithm to optimize gains
do_plots    = 0
