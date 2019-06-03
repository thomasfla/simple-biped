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

def get_test_name(ctrl, zeta, f_dist, w_d4x):
    return ctrl + '_zeta_'+str(zeta)+'_fDist_'+str(f_dist)+'_w_d4x_'+str(w_d4x)
    
def get_gains_file_name(BASE_NAME, w_d4x):
    return BASE_NAME+'_w_d4x='+str(w_d4x)+'.npy'
    
k                   = 1       # contact stiffness multiplier

DATA_DIR                = str(os.path.dirname(os.path.abspath(__file__)))+'/../data/'
TESTS_DIR_NAME          = 'k_'+str(k)+'/push_recovery_slip_ekf_coulomb/'
GAINS_DIR_NAME          = 'gains_k_'+str(k)+'/'
DATA_FILE_NAME          = 'logger_data.npz'
OUTPUT_DATA_FILE_NAME   = 'summary_data'
SAVE_DATA               = 1
LOAD_DATA               = 0
SAVE_FIGURES            = 1

TIME_BETWEEN_TESTS  = 10.0

keys                = ['ctrl', 'fDist', 'zeta', 'w_d4x']
f_dists             = [0.]
T_cost_function     = 10.0
dt_cost_function    = 1e-2  # time step used for cost function of gain tuning
nc                  = 2     # size of CoM vector

# simulation parameters
T_simu              = 6.0
dt_simu             = 1e-3  # time step used by controller
ndt                 = 10    # number of simulation time steps for each control time step
mu                  = 0.3
Ky                  = k*200000. # 23770
Kz                  = k*200000.
K_contact           = np.asmatrix(np.diagflat([Ky,Kz,Ky,Kz]))
zetas               = [0.3]     # contact damping ratio
joint_coulomb_friction = 1*0.4*np.array([1.,10.,1.,10.])    # coulomb friction 'right hip', 'right knee', 'left hip', 'left knee'
JOINT_TORQUES_CUT_FREQUENCY = -30.0
USE_ESTIMATOR       = 1

# INITIAL ROBOT STATE
mass                = 47.21657633
g                   = 9.81
q0 = np.matrix([[ 2.019e-04 , 5.6963e-01 , 1.000e+00, -1.033e-03,  6.799e-04, -1.008e-04,  6.800e-04,  9.811e-05]]).T
v0 = np.matrix([[ 9.470e-02 , 1.234e-05 ,-6.430e-03, -2.164e-01, -8.863e-03, -2.164e-01,  8.950e-03]]).T
# lower base so that initial contact forces compensate for gravity in Z direction
q0[1] -=0.5*mass*g/Kz

w_x         = 1.0
w_dx        = 0e-1 #0.0
w_d2x       = 0e-3
w_d3x       = 0e-6
w_d4x_list  = np.logspace(-6.0, -12.0, num=7)

do_plots    = 0

useViewer   = 0
fdisplay    = 10
camera_transform = [2.31, 0.04, 0.2, 
                    0.5007092356681824, 0.5193040370941162, 0.4947327971458435, 0.48461633920669556]