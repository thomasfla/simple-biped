# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:57:32 2018

Analyze the performance of a controller with different sets of gains.

@author: adelprete
"""

from numpy import matlib
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from simple_biped.utils.logger import RaiLogger
from simple_biped.utils.utils_thomas import compute_stats
from simple_biped.utils.tsid_utils import createContactForceInequalities
from simple_biped.utils.regex_dict import RegexDict
from simple_biped.utils.LDS_utils import simulate_ALDS
from simple_biped.gain_tuning.genetic_tuning import GainOptimizeAdmCtrl, convert_cost_function, compute_projection_to_com_state
from simple_biped.simu import Simu
from simple_biped.admittance_ctrl import GainsAdmCtrl
from simple_biped.hrp2_reduced import Hrp2Reduced
from simple_biped.robot_model_path import pkg, urdf 
import simple_biped.utils.plot_utils as plut
from simple_biped.utils.utils_thomas import finite_diff

import simple_biped.gain_tuning.adm_ctrl_tuning_conf as conf

import os
import itertools
import pickle

class Empty:
    pass    

def compute_stats_and_add_to_data(name, x):
    mean_err, rmse, max_err = compute_stats(x)
    data.__dict__[name+'_mean_err']  = mean_err
    data.__dict__[name+'_rmse']      = rmse
    data.__dict__[name+'_max_err']   = max_err
    data.__dict__[name+'_mse']       = rmse**2
    
np.set_printoptions(precision=1, linewidth=200, suppress=True)


# User parameters
keys            = conf.keys 
controllers     = conf.controllers 
f_dists         = conf.f_dists
zetas           = conf.zetas
T               = conf.T_simu
w_d4x_list      = conf.w_d4x_list

DATA_DIR                = conf.DATA_DIR + conf.TESTS_DIR_NAME
GAINS_DIR               = conf.DATA_DIR + conf.GAINS_DIR_NAME
GAINS_FILE_NAME         = conf.GAINS_FILE_NAME
dt                      = conf.dt_simu
mu                      = conf.mu

DATA_FILE_NAME          = 'logger_data.npz'
OUTPUT_DATA_FILE_NAME   = 'summary_data'
SAVE_DATA               = 1
LOAD_DATA               = 0
plut.SAVE_FIGURES       = 1
SHOW_FIGURES            = 1
plut.FIGURE_PATH        = DATA_DIR

# try to load file containing optimal gains
f = open(GAINS_DIR+GAINS_FILE_NAME+'.pkl', 'rb')
optimal_gains = pickle.load(f)
f.close()

N = int(T/dt)
N_DISTURB_END = 0 #int(conf.T_DISTURB_END/dt) +1
nc = conf.nc
ny = conf.ny
nf = conf.nf

# SETUP
robot   = Hrp2Reduced(urdf, [pkg], loadModel=0, useViewer=0)
q       = robot.q0.copy()
v       = matlib.zeros((robot.model.nv,1))
K       = Simu.get_default_contact_stiffness()
initial_gains = GainsAdmCtrl.get_default_gains(K)
gain_optimizer = GainOptimizeAdmCtrl(robot, q, v, K, ny, nf, initial_gains.to_array(), dt, conf.x0, N, np.eye(2*ny+3*nf))

if(LOAD_DATA):
    try:
        f = open(DATA_DIR+OUTPUT_DATA_FILE_NAME+'.pkl', 'rb');
        res = pickle.load(f); 
        f.close();
    except:
        print "Impossible to open file", DATA_DIR+OUTPUT_DATA_FILE_NAME+'.pkl';
        LOAD_DATA = 0

if(not LOAD_DATA):
    res = RegexDict()

    (B, b) = createContactForceInequalities(mu) # B_f * f <= b_f
    k = B.shape[0]
    B_f = matlib.zeros((2*k,4));
    b_f = matlib.zeros(2*k).T;
    B_f[:k,:2] = B
    B_f[k:,2:] = B
    b_f[:k,0] = b
    b_f[k:,0] = b
    
    time = np.arange(N*dt, step=dt)
    
    Q_state_lin = convert_cost_function(conf.w_x, conf.w_dx, conf.w_d2x, conf.w_d3x, 0.0)
    Q_state_lin_comp = 4*[None]
    Q_state_lin_comp[0] = convert_cost_function(conf.w_x, 0.0, 0.0, 0.0, 0.0)
    Q_state_lin_comp[1] = convert_cost_function(0.0, conf.w_dx, 0.0, 0.0, 0.0)
    Q_state_lin_comp[2] = convert_cost_function(0.0, 0.0, conf.w_d2x, 0.0, 0.0)
    Q_state_lin_comp[3] = convert_cost_function(0.0, 0.0, 0.0, conf.w_d3x, 0.0)
    Q_state = matlib.diagflat(np.matrix(1*[conf.w_x] + 1*[conf.w_dx] + 1*[conf.w_d2x] + 1*[conf.w_d3x]))
    Q_ctrl_lin  = convert_cost_function(0.0, 0.0, 0.0, 0.0, 1.0)
    P = compute_projection_to_com_state()
    P_pinv = np.linalg.pinv(P)
    
    for (ctrl, f_dist, zeta, w_d4x) in itertools.product(controllers, f_dists, zetas, w_d4x_list):

        test_name = conf.get_test_name(ctrl, zeta, f_dist, w_d4x)
        INPUT_FILE = DATA_DIR + test_name + '/' + DATA_FILE_NAME
        
        print '\n'+"".center(120, '#')
        print ("Gonna read %s"%(test_name)).center(120)
        print "".center(120, '#')
            
        # SETUP LOGGER
        data = Empty()
        lgr = RaiLogger()
        try:
            lgr.load(INPUT_FILE)
            
            com_p = lgr.get_vector('simu_com_p', 2)[:,N_DISTURB_END:]
            com_ref = lgr.get_vector('tsid_comref', 2)[:,N_DISTURB_END:]
            com_err = com_p-com_ref
            com_v = lgr.get_vector('simu_com_v', 2)[:,N_DISTURB_END:]
            com_a = lgr.get_vector('simu_com_a', 2)[:,N_DISTURB_END:]
            com_j = lgr.get_vector('simu_com_j', 2)[:,N_DISTURB_END:]
            com_s = np.matrix(finite_diff(com_j, dt, False))
            lkf = lgr.get_vector('simu_lkf', nf/2)[:,N_DISTURB_END:]
            rkf = lgr.get_vector('simu_rkf', nf/2)[:,N_DISTURB_END:]
            f   = np.vstack((lkf,rkf))
    #        df  = lgr.get_vector('simu_df', nf)
    #        ddf = lgr.get_vector('simu_ddf', nf)
            
            # compute state cost and control cost for real system
            N = com_p.shape[1] #len(lgr.get_streams('simu_q_0'))
            time = np.arange(N*dt, step=dt)
            
            state_cost, control_cost, state_cost_component = 0.0, 0.0, np.zeros(4)
            for t in range(N):
                x = np.matrix([[com_err[0,t], com_v[0,t], com_a[0,t], com_j[0,t]]]).T
                u = np.matrix([[com_s[0,t]]])
                state_cost   += dt*(x.T * Q_state * x)[0,0]
                control_cost += dt*(u.T * u)[0,0]
                for i in range(4): state_cost_component[i] += dt*(x[i] * Q_state[i,i] * x[i])
            data.cost_state   = np.sqrt(state_cost/T)
            data.cost_control = np.sqrt(control_cost/T)
            data.cost_state_component = np.sqrt(state_cost_component/T)
            
            fric_cone_viol = np.zeros(N)
            for t in range(f.shape[1]):
                tmp = np.max(B_f * f[:,t] - b_f)
                if(tmp>0.0): fric_cone_viol[t] = tmp
            compute_stats_and_add_to_data('fric_cone_viol', fric_cone_viol)
            
            x0_com = np.vstack((com_err[:,0], com_v[:,0], com_a[:,0], com_j[:,0], 0.0*com_s[:,0]))
            print '1e3*x0_com', 1e3*x0_com.T
            x0 = P_pinv * x0_com
        except:
            print "Could not read file", INPUT_FILE
            data.cost_state   = np.nan
            data.cost_control = np.nan
            data.cost_state_component = 4*[np.nan]        
            x0 = conf.x0
            com_ref = np.nan*matlib.zeros((2, N))
            com_p = np.nan*matlib.zeros((2, N))
            com_v = np.nan*matlib.zeros((2, N))
            com_a = np.nan*matlib.zeros((2, N))
            com_j = np.nan*matlib.zeros((2, N))
            com_s = np.nan*matlib.zeros((2, N))
        
        key = res.generate_key(keys, (ctrl, f_dist, zeta, w_d4x))
        res[key] = data
        
        # simulate expected costs for linear system        
        H = gain_optimizer.compute_transition_matrix(optimal_gains[w_d4x]);
                
        def compute_expected_costs(x):
            x_proj = matlib.empty((P.shape[0], N))  # com state
            expected_state_cost, expected_control_cost, expected_state_cost_component = 0.0, 0.0, np.zeros(4)
            for t in range(N):
                x_proj[:,t] = P * x[:,t]
                expected_state_cost   += dt*(x[:,t].T * Q_state_lin * x[:,t])[0,0]
                expected_control_cost += dt*(x[:,t].T * Q_ctrl_lin * x[:,t])[0,0]
                for i in range(4): expected_state_cost_component[i] += dt*(x[:,t].T * Q_state_lin_comp[i] * x[:,t])
            return x_proj, np.sqrt(expected_state_cost/T), np.sqrt(expected_control_cost/T), np.sqrt(expected_state_cost_component/T)
            
        x = simulate_ALDS(H, x0, dt, N)
        x_proj, data.expected_state_cost, data.expected_control_cost, data.expected_state_cost_component = compute_expected_costs(x)
        
        x2 = simulate_ALDS(H, conf.x0, dt, N)
        x2_proj, data.expected_state_cost2, data.expected_control_cost2, data.expected_state_cost_component2 = compute_expected_costs(x2)
        
        print 'Real state cost:        %f'%(data.cost_state)
        print 'Expected state cost:    %f'%(data.expected_state_cost)
        print 'Real ctrl cost:         %f'%(data.cost_control)
        print 'Expected ctrl cost:     %f'%(data.expected_control_cost)
        for i in range(4):
            print 'Real state component %d cost:        %f'%(i, data.cost_state_component[i])
            print 'Expected state component %d cost:    %f'%(i, data.expected_state_cost_component[i])
        
        def plot_real_vs_expected_com_state(com_expected, name):
            fi, ax = plt.subplots(5, 1, sharex=True);
            i = 0
            ax[i].plot(time, com_expected[0,:].A1, label='expected')
            ax[i].plot(time, (com_p-com_ref)[0,:].A1, '--', label='real')
            ax[i].set_title(r'CoM Pos Y, $w_u$='+str(w_d4x))
            ax[i].legend()
            i += 1
            ax[i].plot(time, com_expected[nc,:].A1, label='expected')
            ax[i].plot(time, com_v[0,:].A1, '--', label='real')
            ax[i].set_title('CoM Vel Y')
            i += 1
            ax[i].plot(time, com_expected[2*nc,:].A1, label='expected')
            ax[i].plot(time, com_a[0,:].A1, '--', label='real')
            ax[i].set_title('Com Acc Y')
            i += 1
            ax[i].plot(time, com_expected[3*nc,:].A1, label='expected')
            ax[i].plot(time, com_j[0,:].A1, '--', label='real')
            ax[i].set_title('Com Jerk Y')
            i += 1
            ax[i].plot(time, com_expected[4*nc,:].A1, label='expected')
            ax[i].plot(time, com_s[0,:].A1, '--', label='real')
            ax[i].set_title('Com Snap Y')
            plut.saveFigure('exp_vs_real_com_Y_'+ctrl+'_w_u_'+str(w_d4x)+'_'+name)
            
        if(plut.SAVE_FIGURES or conf.do_plots):
            # plot real CoM trajectory VS expected CoM trajectory
            plot_real_vs_expected_com_state(x_proj, 'same_x0')
            plot_real_vs_expected_com_state(x2_proj, 'fixed_x0')
            
    if(SAVE_DATA):
        print "Save results in", DATA_DIR + OUTPUT_DATA_FILE_NAME+'.pkl'
        with open(DATA_DIR + OUTPUT_DATA_FILE_NAME+'.pkl', 'wb') as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)

# plot expected costs VS real costs
keys_sorted = optimal_gains.keys()
keys_sorted.sort()
    
plt.figure()
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for (w_d4x, color) in zip(keys_sorted, colors):
    tmp = res.get_matching(keys, [None, None, None, w_d4x]).next()
    plt.plot(tmp.expected_state_cost, tmp.expected_control_cost, ' *', color=color, markersize=30, label=r'Expected, $w_{u}$='+str(w_d4x))
    plt.plot(tmp.cost_state, tmp.cost_control, ' o', color=color, markersize=30) #, label='real w_d4x='+str(w_d4x))
    
plt.legend()
plt.grid(True);
plt.xlabel(r'State cost')
plt.ylabel(r'Control cost')
plut.saveFigure('roc_'+controllers[0]+'_linscale')

plt.xscale('log')
plt.yscale('log')
plut.saveFigure('roc_'+controllers[0]+'_log_scale')


plt.figure()
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for (w_d4x, color) in zip(keys_sorted, colors):
    tmp = res.get_matching(keys, [None, None, None, w_d4x]).next()
    plt.plot(tmp.expected_state_cost2, tmp.expected_control_cost2, ' *', color=color, markersize=30, label=r'Expected, $w_{u}$='+str(w_d4x))
    plt.plot(tmp.cost_state, tmp.cost_control, ' o', color=color, markersize=30) #, label='real w_d4x='+str(w_d4x))
    
plt.legend()
plt.grid(True);
plt.xlabel(r'State cost')
plt.ylabel(r'Control cost')
plut.saveFigure('roc_'+controllers[0]+'_fixed_x0_linscale')

plt.xscale('log')
plt.yscale('log')
plut.saveFigure('roc_'+controllers[0]+'_fixed_x0_log_scale')

if(SHOW_FIGURES):
    plt.show()

