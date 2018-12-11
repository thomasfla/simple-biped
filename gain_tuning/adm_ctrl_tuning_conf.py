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

def get_test_name(ctrl, zeta, f_dist, w_ddf):
    return ctrl + '_zeta_'+str(zeta)+'_fDist_'+str(f_dist)+'_w_ddf_'+str(w_ddf)
    
def get_gains_file_name(w_ddf):
    return GAINS_FILE_NAME+'_w_ddf='+str(w_ddf)+'.npy'
    
DATA_DIR                = str(os.path.dirname(os.path.abspath(__file__)))+'/../data/'
GAINS_DIR_NAME          = 'gains/'
TESTS_DIR_NAME          = 'test_gain_tuning/'
GAINS_FILE_NAME         = 'gains_adm_ctrl'

keys                = ['ctrl', 'fDist', 'zeta', 'w_ddf']
controllers         = ['adm_ctrl']
f_dists             = [400.]
zetas               = [0.3]
T_genetic           = 10.0
T_simu              = 5.0
dt_genetic          = 1e-2  # time step used for gain tuning in genetic algorithm
dt_simu             = 1e-3  # time step used in simulations
mu                  = 0.3
T_DISTURB_BEGIN     = 0.11

w_x         = 1.0
w_dx        = 0.0
w_f         = 0.0
w_df        = 0.0
w_ddf_list  = np.logspace(-16.0, -9.0, num=8)

ny          = 3
nf          = 4
x0          = matlib.zeros((3*nf+2*ny,1))
x0[ny,0]    = .0765     # initial CoM velocity in Y direction

max_iter    = 100        # max number of iterations of genetic algorithm
do_plots    = 0

Q_pos       = matlib.diagflat(np.matrix([w_x,w_x,0.] + ny*[0.] + nf*[0.] + nf*[0.] + nf*[0.0]))
Q_ddf       = matlib.diagflat(np.matrix(ny*[0.0]     + ny*[0.] + nf*[0.] + nf*[0.] + nf*[1.0]))