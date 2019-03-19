# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Configuration for tuning the gains of all controllers

@author: adelprete
"""

#!/usr/bin/env python
from __future__ import print_function

import os
import numpy as np
from numpy import matlib

def get_test_name(ctrl, zeta, f_dist, w_d4x):
    return ctrl + '_zeta_'+str(zeta)+'_fDist_'+str(f_dist)+'_w_d4x_'+str(w_d4x)
    
def get_gains_file_name(BASE_NAME, w_d4x):
    return BASE_NAME+'_w_d4x='+str(w_d4x)+'.npy'
    
DATA_DIR                = str(os.path.dirname(os.path.abspath(__file__)))+'/../data/'
GAINS_DIR_NAME          = 'gains/'
DATA_FILE_NAME          = 'logger_data.npz'
OUTPUT_DATA_FILE_NAME   = 'summary_data'
SAVE_DATA               = 1
LOAD_DATA               = 1
SAVE_FIGURES            = 1

keys                = ['ctrl', 'fDist', 'zeta', 'w_d4x']
f_dists             = [0.]
zetas               = [0.3]
T_cost_function     = 10.0
T_simu              = 2.0
dt_cost_function    = 1e-2  # time step used for cost function of gain tuning
dt_simu             = 1e-3  # time step used in simulations
mu                  = 0.3
nc                  = 2     # size of CoM vector

w_x         = 1.0
w_dx        = 0e-1 #0.0
w_d2x       = 0e-3
w_d3x       = 0e-6
w_d4x_list  = np.logspace(-6.0, -12.0, num=7)
#w_d4x_list  = np.hstack((np.logspace(-10.0, -7.0, num=4)) , [2.5, 5.0]))

do_plots    = 0
