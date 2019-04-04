# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Tune the gains for TSID-Admittance using LQR and saves the gains in binary files.

@author: adelprete
"""

#!/usr/bin/env python
from __future__ import print_function

import os
import time
import pickle
import numpy as np
from numpy import matlib
from numpy.linalg import eigvals
import matplotlib.pyplot as plt
import simple_biped.utils.plot_utils as plut

from simple_biped.simu import Simu
from simple_biped.utils.LDS_utils import simulate_ALDS, compute_integrator_dynamics, compute_weighted_quadratic_state_control_integral_ALDS, compute_integrator_gains
from simple_biped.tsid_admittance import GainsTsidAdm
from simple_biped.gain_tuning.tune_gains_tsid_adm_utils import convert_integrator_gains_to_tsid_adm_gains, convert_tsid_adm_gains_to_integrator_gains
import controlpy

import simple_biped.gain_tuning.conf_tsid_adm as conf

DATA_DIR                = conf.DATA_DIR + conf.GAINS_DIR_NAME
OUTPUT_DATA_FILE_NAME   = conf.GAINS_FILE_NAME # 'gains_adm_ctrl'
SAVE_DATA               = 1
LOAD_DATA               = 0 # if 1 it tries to load the gains from the specified binary file

N           = int(conf.T_cost_function/conf.dt_cost_function)
dt          = conf.dt_cost_function
w_x         = conf.w_x
w_dx        = conf.w_dx
w_d2x       = conf.w_d2x
w_d3x       = conf.w_d3x
w_d4x_list  = conf.w_d4x_list
x0          = conf.x0
do_plots    = conf.do_plots         # if true it shows the plots

K_contact = conf.K_contact
(H, A, B) = compute_integrator_dynamics(matlib.zeros((1,4)))
Q = matlib.diagflat([w_x, w_dx, w_d2x, w_d3x])
R = matlib.diagflat([1.0])

if(LOAD_DATA):
    try:
        f = open(DATA_DIR+OUTPUT_DATA_FILE_NAME+'.pkl', 'rb')
        optimal_gains = pickle.load(f)
        f.close()
    except:
        print("Impossible to open file", DATA_DIR+OUTPUT_DATA_FILE_NAME+'.pkl')
        LOAD_DATA = 0

if(not LOAD_DATA):
    optimal_gains = {}
    
    start_time = time.time()
    for w_d4x in w_d4x_list:
        print("".center(100,'#'))
        print("Tuning gains for w_d4x={}".format(w_d4x))

        if(w_d4x<1.0):
            R[0,0] = w_d4x
            gains_4th_order_system, X, closedLoopEigVals = controlpy.synthesis.controller_lqr(A,B,Q,R)
        else:
            # TEMP
            p1, dp =-5.0, -w_d4x
            gains_4th_order_system = compute_integrator_gains(4, p1, dp)
            #END TEMP
        
        tmp = convert_integrator_gains_to_tsid_adm_gains(gains_4th_order_system, K_contact)
        optimal_gains[w_d4x] = tmp.to_array()
        
        print("Optimal gains:\n", tmp.to_string())
    
    print("Total time taken to optimize gains", time.time()-start_time)
    
    if(SAVE_DATA):
        print("Save results in", DATA_DIR + OUTPUT_DATA_FILE_NAME+'.pkl')
        try:
            os.makedirs(DATA_DIR);
        except OSError:
            print("Directory already exists.")
            
        with open(DATA_DIR + OUTPUT_DATA_FILE_NAME+'.pkl', 'wb') as f:
            pickle.dump(optimal_gains, f, pickle.HIGHEST_PROTOCOL)
        
        for w_d4x in w_d4x_list:
            with open(DATA_DIR + conf.get_gains_file_name(conf.GAINS_FILE_NAME, w_d4x), 'wb') as f:
                np.save(f, optimal_gains[w_d4x])
            

# SETUP
#from .analyze_common import compute_cost_function_matrix 
#
#ny          = conf.ny
#Q_pos_sqrt  = np.sqrt(conf.Q_pos)
#R_pos_sqrt  = np.sqrt(conf.R_pos)
#Q_d4x_sqrt  = np.sqrt(conf.Q_d4x)
#R_d4x_sqrt  = np.sqrt(conf.R_d4x)
#Q_sqrt      = np.sqrt(conf.Q_pos)
#R_sqrt      = matlib.zeros((ny,ny))
#T           = conf.T_cost_function
#optimal_cost_pos = {}
#optimal_cost_d4x = {}
#keys_sorted = optimal_gains.keys()
#keys_sorted.sort()
#
#for w_d4x in keys_sorted:
#    gains = GainsTsidAdm(optimal_gains[w_d4x])
#    K = convert_tsid_adm_gains_to_integrator_gains(gains, K_contact)
#
#    R_sqrt[0,0] = np.sqrt(w_d4x)    
#    optimal_cost               = compute_weighted_quadratic_state_control_integral_ALDS(A, B, K, x0, T, Q_sqrt, R_sqrt, dt)
#    optimal_cost_pos[w_d4x]    = compute_weighted_quadratic_state_control_integral_ALDS(A, B, K, x0, T, Q_pos_sqrt, R_pos_sqrt, dt)
#    optimal_cost_d4x[w_d4x]    = compute_weighted_quadratic_state_control_integral_ALDS(A, B, K, x0, T, Q_d4x_sqrt, R_d4x_sqrt, dt)
#
#    H = A - B*K  # closed-loop transition matrix    
#    print("".center(100,'#'))
#    print("w_d4x={}".format(w_d4x))
#    print("Optimal cost     {}".format(optimal_cost))
#    print("Optimal cost state {}".format(optimal_cost_pos[w_d4x]))
#    print("Optimal cost ctrl  {}".format(optimal_cost_d4x[w_d4x]))
#    print("Largest eigenvalues:", np.sort_complex(eigvals(H))[-4:].T)
#    
#    if(do_plots):
#        simulate_ALDS(H, x0, dt, N, 1, 0)
#        plt.title("log(w_d4x)=%.1f"%(np.log10(w_d4x)))
#
#plt.figure()
#for w_d4x in keys_sorted:
#    plt.plot(optimal_cost_pos[w_d4x], optimal_cost_d4x[w_d4x], ' *', markersize=30, label='w_d4x='+str(w_d4x))
#plt.legend()
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel('State tracking cost')
#plt.ylabel('Control cost')
#plt.grid(True);
#
#if(do_plots):    
#    plt.show()
