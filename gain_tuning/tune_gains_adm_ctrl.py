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
from simple_biped.gain_tuning.genetic_tuning import optimize_gains_adm_ctrl, GainOptimizeAdmCtrl
from simple_biped.hrp2_reduced import Hrp2Reduced
from simple_biped.robot_model_path import pkg, urdf 

import simple_biped.gain_tuning.adm_ctrl_tuning_conf as conf

DATA_DIR                = conf.DATA_DIR + conf.GAINS_DIR_NAME
OUTPUT_DATA_FILE_NAME   = conf.GAINS_FILE_NAME # 'gains_adm_ctrl'
SAVE_DATA               = 1
LOAD_DATA               = 0 # if 1 it tries to load the gains from the specified binary file

N           = int(conf.T_genetic/conf.dt_genetic)
dt          = conf.dt_genetic
w_x         = conf.w_x
w_dx        = conf.w_dx
w_f         = conf.w_f
w_df        = conf.w_df
w_ddf_list  = conf.w_ddf_list
ny          = conf.ny
nf          = conf.nf
x0          = conf.x0
max_iter    = conf.max_iter        # max number of iteration of genetic algorithm
do_plots    = conf.do_plots         # if true it shows the plots

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
    for w_ddf in w_ddf_list:
        print("".center(100,'#'))
        print("Tuning gains for w_ddf={}".format(w_ddf))
        optimal_gains[w_ddf] = optimize_gains_adm_ctrl(w_x, w_dx, w_f, w_df, w_ddf, N, dt, max_iter, x0, initial_guess, do_plots=0)
        
        # update initial guess
        initial_guess = optimal_gains[w_ddf]
        print("Optimal gains:\n", GainsAdmCtrl(optimal_gains[w_ddf]).to_string())
    
    print("Total time taken to optimize gains", time.time()-start_time)
    
    if(SAVE_DATA):
        print("Save results in", DATA_DIR + OUTPUT_DATA_FILE_NAME+'.pkl')
        try:
            os.makedirs(DATA_DIR);
        except OSError:
            print("Directory already exists.")
            
        with open(DATA_DIR + OUTPUT_DATA_FILE_NAME+'.pkl', 'wb') as f:
            pickle.dump(optimal_gains, f, pickle.HIGHEST_PROTOCOL)
        
        for w_ddf in w_ddf_list:
            with open(DATA_DIR + conf.get_gains_file_name(w_ddf), 'wb') as f:
                np.save(f, optimal_gains[w_ddf])
            

# SETUP
robot   = Hrp2Reduced(urdf, [pkg], loadModel=0, useViewer=0)
q       = robot.q0.copy()
v       = matlib.zeros((robot.model.nv,1))

Q_pos   = conf.Q_pos #matlib.diagflat(np.matrix(ny*[w_x] + ny*[w_dx] + nf*[w_f] + nf*[w_df] + nf*[0.0]))
Q_ddf   = conf.Q_ddf #matlib.diagflat(np.matrix(ny*[0.0] + ny*[w_dx] + nf*[w_f] + nf*[w_df] + nf*[1.0]))

gain_optimizer = GainOptimizeAdmCtrl(robot, q, v, K, ny, nf, initial_gains.to_array(), dt, x0, N, Q_pos)
normalized_initial_gains = matlib.ones_like(initial_gains.to_array())
gain_optimizer.set_cost_function_matrix(Q_pos)
initial_cost_pos    = gain_optimizer.cost_function(normalized_initial_gains)
gain_optimizer.set_cost_function_matrix(Q_ddf)
initial_cost_ddf    = gain_optimizer.cost_function(normalized_initial_gains)

if(do_plots):
    H = gain_optimizer.compute_transition_matrix(initial_gains.to_array());
    simulate_ALDS(H, x0, dt, N, 1, 0)
    plt.title("Initial gains")

optimal_cost_pos = {}
optimal_cost_ddf = {}
keys_sorted = optimal_gains.keys()
keys_sorted.sort()
for w_ddf in keys_sorted:
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

    H = gain_optimizer.compute_transition_matrix(gains);
    print("Largest eigenvalues:", np.sort_complex(eigvals(H))[-4:].T)
    
    if(do_plots):
        simulate_ALDS(H, x0, dt, N, 1, 0)
        plt.title("log(w_ddf)=%.1f"%(np.log10(w_ddf)))

plt.figure()
for w_ddf in keys_sorted:
    plt.plot(optimal_cost_pos[w_ddf], optimal_cost_ddf[w_ddf], ' *', markersize=30, label='w_ddf='+str(w_ddf))
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Position tracking cost')
plt.ylabel('Force acceleration cost')
plt.grid(True);

if(do_plots):    
    plt.show()