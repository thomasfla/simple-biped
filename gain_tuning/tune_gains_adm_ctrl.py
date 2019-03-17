# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Tune the gains for Admittance Control using a genetic algorithm
and saves the gains in binary files.

@author: adelprete
"""

#!/usr/bin/env python
from __future__ import print_function
import itertools
from select     import select
from subprocess import Popen, PIPE, STDOUT

import os
import time
import pickle
import numpy as np
from numpy import matlib
from numpy.linalg import eigvals
import matplotlib.pyplot as plt
import simple_biped.utils.plot_utils as plut

from simple_biped.admittance_ctrl import GainsAdmCtrl
from simple_biped.simu import Simu
from simple_biped.utils.LDS_utils import simulate_ALDS
from simple_biped.gain_tuning.tune_gains_adm_ctrl_utils import optimize_gains_adm_ctrl, convert_cost_function, GainOptimizeAdmCtrl, compute_projection_to_com_state
from simple_biped.hrp2_reduced import Hrp2Reduced
from simple_biped.robot_model_path import pkg, urdf 

import simple_biped.gain_tuning.conf_adm_ctrl as conf

DATA_DIR                = conf.DATA_DIR + conf.GAINS_DIR_NAME
OUTPUT_DATA_FILE_NAME   = conf.GAINS_FILE_NAME # 'gains_adm_ctrl'
SAVE_DATA               = 1
LOAD_DATA               = 0 # if 1 it tries to load the gains from the specified binary file
N                       = int(conf.T_genetic/conf.dt_genetic)
dt                      = conf.dt_genetic
w_d4x_list              = conf.w_d4x_list
x0                      = conf.x0
plut.SAVE_FIGURES       = 1
plut.FIGURE_PATH        = DATA_DIR

#P = compute_projection_to_com_state()
#x0 = np.linalg.pinv(P)*conf.x0_com
print("x0:\n", x0.T)

K = Simu.get_default_contact_stiffness()
initial_gains = GainsAdmCtrl.get_default_gains(K)
initial_guess = initial_gains.to_array()
print("Initial gains:\n", initial_gains.to_string())

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
        Q = convert_cost_function(conf.w_x, conf.w_dx, conf.w_d2x, conf.w_d3x, w_d4x)
        optimal_gains[w_d4x] = optimize_gains_adm_ctrl(Q, N, dt, conf.max_iter, x0, initial_guess, do_plots=0)
        
        # update initial guess
        initial_guess = optimal_gains[w_d4x]
        print("Optimal gains:\n", GainsAdmCtrl(optimal_gains[w_d4x]).to_string())
    
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
            with open(DATA_DIR + conf.get_gains_file_name(w_d4x), 'wb') as f:
                np.save(f, optimal_gains[w_d4x])
            

# SETUP
robot   = Hrp2Reduced(urdf, [pkg], loadModel=0, useViewer=0)
q       = robot.q0.copy()
v       = matlib.zeros((robot.model.nv,1))

Q_pos   = convert_cost_function(conf.w_x, conf.w_dx, conf.w_d2x, conf.w_d3x, 0.0)
Q_ddf   = convert_cost_function(0.0, 0.0, 0.0, 0.0, 1.0)

gain_optimizer = GainOptimizeAdmCtrl(robot, q, v, K, conf.ny, conf.nf, initial_gains.to_array(), dt, x0, N, Q_pos)
normalized_initial_gains = matlib.ones_like(initial_gains.to_array())
gain_optimizer.set_cost_function_matrix(Q_pos)
initial_cost_pos    = gain_optimizer.cost_function(normalized_initial_gains)
gain_optimizer.set_cost_function_matrix(Q_ddf)
initial_cost_ddf    = gain_optimizer.cost_function(normalized_initial_gains)

if(conf.do_plots):
    H = gain_optimizer.compute_transition_matrix(initial_gains.to_array());
    simulate_ALDS(H, x0, dt, N, 1, 0)
    plt.title("Initial gains")

optimal_cost_pos = {}
optimal_cost_ddf = {}
keys_sorted = optimal_gains.keys()
keys_sorted.sort()
print("Initial cost pos {}".format(initial_cost_pos))
print("Initial cost ddf {}".format(initial_cost_ddf))

for w_d4x in keys_sorted:
    gains = optimal_gains[w_d4x]
    normalized_opt_gains = gain_optimizer.normalize_gains_array(gains)
    
#    Q = convert_cost_function(conf.w_x, conf.w_dx, conf.w_d2x, conf.w_d3x, w_d4x)
#    gain_optimizer.set_cost_function_matrix(Q)
#    initial_cost    = gain_optimizer.cost_function(normalized_initial_gains)
#    optimal_cost    = gain_optimizer.cost_function(normalized_opt_gains)
    gain_optimizer.set_cost_function_matrix(Q_pos)
    optimal_cost_pos[w_d4x]    = gain_optimizer.cost_function(normalized_opt_gains)
    gain_optimizer.set_cost_function_matrix(Q_ddf)
    optimal_cost_ddf[w_d4x]    = gain_optimizer.cost_function(normalized_opt_gains)
    
    
    print("".center(100,'#'))
    print("w_d4x={}".format(w_d4x))
#    print("Initial cost     {}".format(initial_cost))
    
#    print("Optimal cost     {}".format(optimal_cost))    
    print("Optimal cost state {}".format(optimal_cost_pos[w_d4x]))    
    print("Optimal cost ctrl  {}".format(optimal_cost_ddf[w_d4x]))

    H = gain_optimizer.compute_transition_matrix(gains);
    print("Largest eigenvalues:", np.sort_complex(eigvals(H))[-4:].T)
    
    if(conf.do_plots):
        simulate_ALDS(H, x0, dt, N, 1, 0)
        plt.title("log(w_d4x)=%.1f"%(np.log10(w_d4x)))

plt.figure()
for w_d4x in keys_sorted:
    plt.plot(optimal_cost_pos[w_d4x], optimal_cost_ddf[w_d4x], ' *', markersize=30, label='w_d4x=%.1e'%(w_d4x))
plt.legend()
plt.xlabel('State tracking cost')
plt.ylabel('Control cost')
plt.grid(True);
plut.saveFigure('roc_adm_ctrl_lin_scale')

plt.xscale('log')
plt.yscale('log')
plut.saveFigure('roc_adm_ctrl_log_scale')

if(conf.do_plots):    
    plt.show()