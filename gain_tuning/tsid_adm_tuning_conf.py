# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Configuration for tuning the gains for TSID-Admittance

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
TESTS_DIR_NAME          = 'test_gain_tuning/tsid_amd_w_dx_0.1_d2x_1e-3_d3x_1e-6/'
GAINS_FILE_NAME         = 'gains_tsid_adm'

keys                = ['ctrl', 'fDist', 'zeta', 'w_d4x']
controllers         = ['tsid_adm']
f_dists             = [0.]
zetas               = [0.3]
T_cost_function     = 10.0
T_simu              = 5.0
dt_cost_function    = 1e-2  # time step used for cost function of gain tuning
dt_simu             = 1e-3  # time step used in simulations
mu                  = 0.3
T_DISTURB_BEGIN     = 0.0

w_x         = 1.0
w_dx        = 1e-1 #0.0
w_d2x       = 1e-3
w_d3x       = 1e-6
w_d4x_list  = np.logspace(-12.0, -6.0, num=7)
#w_d4x_list  = np.hstack((np.logspace(-10.0, -7.0, num=4)) , [2.5, 5.0]))

ny          = 1
nf          = 4
x0          = matlib.zeros((4*ny,1))
x0[1,0]    = .1     # initial CoM velocity in Y direction

do_plots    = 0

Q_pos       = matlib.diagflat(np.matrix(ny*[w_x] + ny*[w_dx] + ny*[w_d2x] + ny*[w_d3x]))
Q_d4x       = matlib.diagflat(np.matrix(ny*[0.0] + ny*[0.] + ny*[0.] + ny*[0.]))
R_pos       = matlib.diagflat(np.matrix(ny*[0.0]))
R_d4x       = matlib.diagflat(np.matrix(ny*[1.0]))