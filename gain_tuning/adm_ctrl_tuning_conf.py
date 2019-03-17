# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Configuration for tuning the gains for Admittance Control

@author: adelprete
"""

#!/usr/bin/env python
from __future__ import print_function

import os
import numpy as np
from numpy import matlib

def get_test_name(ctrl, zeta, f_dist, w_d4x):
    return ctrl + '_zeta_'+str(zeta)+'_fDist_'+str(f_dist)+'_w_d4x_'+str(w_d4x)
    
def get_gains_file_name(w_d4x):
    return GAINS_FILE_NAME+'_w_d4x='+str(w_d4x)+'.npy'
    
DATA_DIR                = str(os.path.dirname(os.path.abspath(__file__)))+'/../data/'
GAINS_DIR_NAME          = 'gains/'
TESTS_DIR_NAME          = 'test_gain_tuning/adm_ctrl_w_dx_0_d2x_0_d3x_0/'
GAINS_FILE_NAME         = 'gains_adm_ctrl'

keys                = ['ctrl', 'fDist', 'zeta', 'w_ddf']
controllers         = ['adm_ctrl']
f_dists             = [0.]
zetas               = [0.3]
T_genetic           = 10.0
T_simu              = 5.0
dt_genetic          = 1e-2  # time step used for gain tuning in genetic algorithm
dt_simu             = 1e-3  # time step used in simulations
mu                  = 0.3
T_DISTURB_END       = 0.0

w_x         = 1.0
#w_ddf_list  = np.logspace(-16.0, -9.0, num=8)
w_x         = 1.0
w_dx        = 0e-1 #0.0
w_d2x       = 0e-3
w_d3x       = 0e-6
#w_d4x_list  = np.logspace(-12.0, -7.0, num=6)
w_d4x_list  = np.logspace(-6.0, -12.0, num=7)

nc          = 2     # size of CoM vector
ny          = 3
nf          = 4

# state used to compute expected trajectory to compare to real traj (fixed_x0)
x0          = matlib.zeros((3*nf+2*ny,1))
x0[ny,0]    = 47.2 * .1     # initial CoM velocity in Y direction times robot mass

# state used for gain tuning, which is equal to initial simulation state
x0_com      = 1e-3*np.matrix([[ 0.0, 0.0, 1.1e+02, -7.5e-02, 6.0e+01, -3.0e+02, -2.8e-11, 2.9e-11, 0.0, 0.0]]).T #2.6e+08, 6.7e+07]]).T

max_iter    = 1        # max number of iterations of genetic algorithm
do_plots    = 1
