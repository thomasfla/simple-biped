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
import matplotlib.pyplot as plt
import simple_biped.utils.plot_utils as plut

from simple_biped.admittance_ctrl import GainsAdmCtrl
from simple_biped.simu import Simu
from simple_biped.utils.LDS_utils import simulate_ALDS
from simple_biped.gain_tuning.genetic_tuning import optimize_gains_adm_ctrl, GainOptimizeAdmCtrl
from simple_biped.hrp2_reduced import Hrp2Reduced
from simple_biped.robot_model_path import pkg, urdf 

DATA_DIR                = os.getcwd()+'/../data/gains/'
OUTPUT_DATA_FILE_NAME   = 'gains_adm_ctrl'
SAVE_DATA               = 1
LOAD_DATA               = 1 # if 1 it tries to load the gains from the specified binary file

controller  = 'adm_ctrl'
N           = 400
dt          = 1e-2
w_x         = 1.0
w_dx        = 0.0
w_f         = 0.0
w_df        = 0.0
w_ddf_list  = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6] #[1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
max_iter    = 1
do_plots    = 1

K = Simu.get_default_contact_stiffness()
initial_gains = GainsAdmCtrl.get_default_gains(K)
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
    for w_ddf in w_ddf_list:
        print("".center(100,'#'))
        print("Tuning gains for w_ddf={}".format(w_ddf))
        optimal_gains[w_ddf] = optimize_gains_adm_ctrl(w_x, w_dx, w_f, w_df, w_ddf, N, dt, max_iter, do_plots=0)
        print("Optimal gains:\n", GainsAdmCtrl(optimal_gains[w_ddf]).to_string())
    
    if(SAVE_DATA):
        print("Save results in", DATA_DIR + OUTPUT_DATA_FILE_NAME+'.pkl')
        with open(DATA_DIR + OUTPUT_DATA_FILE_NAME+'.pkl', 'wb') as f:
            pickle.dump(optimal_gains, f, pickle.HIGHEST_PROTOCOL)
        
        for w_ddf in w_ddf_list:
            with open(DATA_DIR + OUTPUT_DATA_FILE_NAME+'_w_ddf='+str(w_ddf)+'.npy', 'wb') as f:
                np.save(f, optimal_gains[w_ddf])
            
ny = 3
nf = 4
ss = 3*nf+2*ny
x0 = matlib.zeros((ss,1))
x0[0,0] = 1.0

# SETUP
robot   = Hrp2Reduced(urdf, [pkg], loadModel=0, useViewer=0)
q       = robot.q0.copy()
v       = matlib.zeros((robot.model.nv,1))

Q       = matlib.diagflat(np.matrix(ny*[w_x] + ny*[w_dx] + nf*[w_f] + nf*[w_df] + nf*[w_ddf_list[0]]))
Q_pos   = matlib.diagflat(np.matrix(ny*[w_x] + ny*[w_dx] + nf*[w_f] + nf*[w_df] + nf*[0.0]))
Q_ddf   = matlib.diagflat(np.matrix(ny*[0.0] + ny*[w_dx] + nf*[w_f] + nf*[w_df] + nf*[w_ddf_list[0]]))

gain_optimizer = GainOptimizeAdmCtrl(robot, q, v, K, ny, nf, initial_gains.to_array(), dt, x0, N, Q)
normalized_initial_gains = matlib.ones_like(initial_gains.to_array())
gain_optimizer.set_cost_function_matrix(Q_pos)
initial_cost_pos    = gain_optimizer.cost_function(normalized_initial_gains)
gain_optimizer.set_cost_function_matrix(Q_ddf)
initial_cost_ddf    = gain_optimizer.cost_function(normalized_initial_gains)

if(do_plots):
    H = gain_optimizer.compute_transition_matrix(initial_gains.to_array());
    simulate_ALDS(H, x0, dt, N, 1, 0)
    plt.title("Reponse with initial gains")

optimal_cost_pos = {}
optimal_cost_ddf = {}
for w_ddf in optimal_gains.keys():
    gains = optimal_gains[w_ddf]
    normalized_opt_gains = gain_optimizer.normalize_gains_array(gains)
    
    Q       = matlib.diagflat(np.matrix(ny*[w_x] + ny*[w_dx] + nf*[w_f] + nf*[w_df] + nf*[w_ddf]))
    gain_optimizer.set_cost_function_matrix(Q)
    initial_cost    = gain_optimizer.cost_function(normalized_initial_gains)
    optimal_cost    = gain_optimizer.cost_function(normalized_opt_gains)
    gain_optimizer.set_cost_function_matrix(Q_pos)
    optimal_cost_pos[w_ddf]    = gain_optimizer.cost_function(normalized_opt_gains)
    gain_optimizer.set_cost_function_matrix(Q_ddf)
    optimal_cost_ddf[w_ddf]    = gain_optimizer.cost_function(normalized_opt_gains)
    
    print("".center(100,'#'))
    print("w_ddf={}".format(w_ddf))
    print("Initial cost     {}".format(initial_cost))
    print("Optimal cost     {}".format(optimal_cost))
    print("Initial cost pos {}".format(initial_cost_pos))
    print("Optimal cost pos {}".format(optimal_cost_pos[w_ddf]))
    print("Initial cost ddf {}".format(initial_cost_ddf))
    print("Optimal cost ddf {}".format(optimal_cost_ddf[w_ddf]))
    
    if(do_plots):
        H = gain_optimizer.compute_transition_matrix(gains);
        simulate_ALDS(H, x0, dt, N, 1, 0)
        plt.title("log(w_ddf)=%.1f"%(np.log10(w_ddf)))

plt.figure()
for w_ddf in optimal_gains.keys():
    plt.plot(optimal_cost_pos[w_ddf], optimal_cost_ddf[w_ddf], ' *', markersize=30, label='w_ddf='+str(w_ddf))
plt.legend()
plt.xlabel('Position tracking cost')
plt.ylabel('Force acceleration cost')
plt.grid(True);

if(do_plots):    
    plt.show()