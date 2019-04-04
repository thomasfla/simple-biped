# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Tune the gains for TSID-Rigid and saves the gains in binary files.

@author: adelprete
"""

#!/usr/bin/env python
from __future__ import print_function

import os
import time
import pickle
import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import simple_biped.utils.plot_utils as plut

from simple_biped.utils.LDS_utils import compute_integrator_dynamics
import controlpy
from pinocchio_inv_dyn.first_order_low_pass_filter import FirstOrderLowPassFilter

import simple_biped.gain_tuning.conf_tsid_rigid as conf

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

(H, A, B) = compute_integrator_dynamics(matlib.zeros((1,2)))
Q = matlib.diagflat([w_x, w_dx])
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
        print("".center(60,'#'))
        print("Tuning gains for w_d4x={}".format(w_d4x))
        R[0,0] = w_d4x
        gains_2nd_order_system, X, closedLoopEigVals = controlpy.synthesis.controller_lqr(A,B,Q,R)
        optimal_gains[w_d4x] = gains_2nd_order_system.A1
        print("Optimal gains:\n", gains_2nd_order_system)
    
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
 

def simulate_LDS_with_LPF(A, B, K, x0, DT, N, ndt, fc, plot):
    '''Simulate a Linear Dynamical System (LDS) forward in time assuming
       the control inputs are low-pass filtered
        A: transition matrix
        B: control matrix
        K: feedback gain matrix
        x0: initial state
        DT: time step
        N: number of time steps to simulate
        ndt: number of sub time steps (to make simulation more accurate)
    '''
    n = A.shape[0]
    m = B.shape[1]
    x = matlib.empty((n,N))
    u = matlib.empty((m,N-1))
    u_filt = matlib.empty((m,N-1))
    x[:,0] = x0
    u[:,0] = -K*x0
    u_filt[:,0] = -K*x0
    dt = DT/ndt
    lpf = FirstOrderLowPassFilter(dt, fc, u[:,0].A1)    
    for i in range(N-1):
        u[:,i] = -K*x[:,i]
        x_pre = x[:,i]
        for ii in range(ndt):
            u_filt[:,i] = np.matrix(lpf.filter_data(u[:,i].A1))
            x_post = x_pre + dt * (A*x_pre + B*u_filt[:,i])
            x_pre = x_post
        x[:,i+1] = x_post
        
    if plot:
        max_rows = 4
        n_cols = 1 + (n+m+1)/max_rows
        n_rows = int(np.ceil(float(n+m)/n_cols))
        f, ax = plt.subplots(n_rows, n_cols, sharex=True);
        ax = ax.reshape(n_cols*n_rows)
        time = np.arange(N*DT, step=DT)
        for i in range(n):
            ax[i].plot(time, x[i,:].A1)
            ax[i].set_title('x '+str(i))
        for i in range(m):
            ax[n+i].plot(time[:-1], u[i,:].A1, label='u')
            ax[n+i].plot(time[:-1], u_filt[i,:].A1, '--', label='u filtered')
            ax[n+i].set_title('u '+str(i))
            ax[n+1].legend(loc='best')
    return (x,u)           

# SETUP
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
keys_sorted = optimal_gains.keys()
keys_sorted.sort()
x0 = x0[::2]
fc = 30
dt = 1e-3
ndt = 10
N  = int(conf.T_simu/dt)
for w_d4x in keys_sorted:
    K = optimal_gains[w_d4x]

#    R_sqrt[0,0] = np.sqrt(w_d4x)    
#    optimal_cost               = compute_weighted_quadratic_state_control_integral_ALDS(A, B, K, x0, T, Q_sqrt, R_sqrt, dt)
#    optimal_cost_pos[w_d4x]    = compute_weighted_quadratic_state_control_integral_ALDS(A, B, K, x0, T, Q_pos_sqrt, R_pos_sqrt, dt)
#    optimal_cost_d4x[w_d4x]    = compute_weighted_quadratic_state_control_integral_ALDS(A, B, K, x0, T, Q_d4x_sqrt, R_d4x_sqrt, dt)

    H = A - B*K  # closed-loop transition matrix    
    print("".center(60,'#'))
    print("w_d4x={}".format(w_d4x))
#    print("Optimal cost     {}".format(optimal_cost))
#    print("Optimal cost state {}".format(optimal_cost_pos[w_d4x]))
#    print("Optimal cost ctrl  {}".format(optimal_cost_d4x[w_d4x]))
#    print("Largest eigenvalues:", np.sort_complex(eigvals(H))[-4:].T)
    
    if(do_plots):
        simulate_LDS_with_LPF(A, B, K, x0, dt, N, ndt, fc, 1)
        plt.title("log(w_d4x)=%.1f"%(np.log10(w_d4x)))

#plt.figure()
#for w_d4x in keys_sorted:
#    plt.plot(optimal_cost_pos[w_d4x], optimal_cost_d4x[w_d4x], ' *', markersize=30, label='w_d4x='+str(w_d4x))
#plt.legend()
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel('State tracking cost')
#plt.ylabel('Control cost')
#plt.grid(True);

if(do_plots):    
    plt.show()
