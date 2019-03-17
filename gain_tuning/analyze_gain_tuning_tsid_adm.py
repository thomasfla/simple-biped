# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:57:32 2018

Analyze the performance of TSID-Admittance with different sets of gains.

@author: adelprete
"""

from numpy import matlib
import numpy as np
from numpy.linalg import eigvals
import matplotlib.pyplot as plt

from simple_biped.utils.logger import RaiLogger
from simple_biped.utils.utils_thomas import compute_stats, finite_diff
from simple_biped.utils.tsid_utils import createContactForceInequalities
from simple_biped.utils.regex_dict import RegexDict
from simple_biped.utils.LDS_utils import simulate_LDS, compute_integrator_dynamics
from simple_biped.tsid_admittance import GainsTsidAdm
from simple_biped.gain_tuning.tune_gains_tsid_adm_utils import convert_tsid_adm_gains_to_integrator_gains
from simple_biped.simu import Simu
import simple_biped.utils.plot_utils as plut

import simple_biped.gain_tuning.conf_tsid_adm as conf

import os
import itertools
import pickle

class Empty:
    pass    
    
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
N_DISTURB_BEGIN = 0 #int(conf.T_DISTURB_BEGIN/dt) + 1
ny = conf.ny
nf = conf.nf
x0 = conf.x0

# SETUP
K_contact       = Simu.get_default_contact_stiffness()
(H, A, B) = compute_integrator_dynamics(matlib.zeros((1,4)))

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

    (B_tmp, b_tmp) = createContactForceInequalities(mu) # B_f * f <= b_f
    k = B_tmp.shape[0]
    B_f = matlib.zeros((2*k,4));
    b_f = matlib.zeros(2*k).T;
    B_f[:k,:2] = B_tmp
    B_f[k:,2:] = B_tmp
    b_f[:k,0] = b_tmp
    b_f[k:,0] = b_tmp
    
    time = np.arange(N*dt, step=dt)
    
    for (ctrl, f_dist, zeta, w_d4x) in itertools.product(controllers, f_dists, zetas, w_d4x_list):

        test_name = conf.get_test_name(ctrl, zeta, f_dist, w_d4x)
        INPUT_FILE = DATA_DIR + test_name + '/' + DATA_FILE_NAME
        
        print '\n'+"".center(120, '#')
        print ("Gonna read %s"%(test_name)).center(120)
        print "".center(120, '#')
            
        # SETUP LOGGER
        lgr = RaiLogger()
        try:
            lgr.load(INPUT_FILE)
        except:
            print "Could not read file", INPUT_FILE
            continue
        
        nf = 4
        com_p = lgr.get_vector('simu_com_p', 2)[:,N_DISTURB_BEGIN:]
        com_ref = lgr.get_vector('tsid_comref', 2)[:,N_DISTURB_BEGIN:]
        com_err = com_p-com_ref
        com_v = lgr.get_vector('simu_com_v', 2)[:,N_DISTURB_BEGIN:]
        com_a = lgr.get_vector('simu_com_a', 2)[:,N_DISTURB_BEGIN:]
        com_j = lgr.get_vector('simu_com_j', 2)[:,N_DISTURB_BEGIN:]
        com_s = np.matrix(finite_diff(com_j, dt, False))
        lkf = lgr.get_vector('simu_lkf', nf/2)[:,N_DISTURB_BEGIN:]
        rkf = lgr.get_vector('simu_rkf', nf/2)[:,N_DISTURB_BEGIN:]
        f   = np.vstack((lkf,rkf))
#        df  = lgr.get_vector('simu_df', nf)[:,N_DISTURB_BEGIN:]
#        ddf = lgr.get_vector('simu_ddf', nf)[:,N_DISTURB_BEGIN:]
        
        data = Empty()
        def compute_stats_and_add_to_data(name, x):
            mean_err, rmse, max_err = compute_stats(x)
            data.__dict__[name+'_mean_err']  = mean_err
            data.__dict__[name+'_rmse']      = rmse
            data.__dict__[name+'_max_err']   = max_err
            data.__dict__[name+'_mse']       = rmse**2
        
        # compute state cost
        N = com_p.shape[1] #len(lgr.get_streams('simu_q_0'))
        time = np.arange(N*dt, step=dt)
        
        state_cost, control_cost, state_cost_component = 0.0, 0.0, np.zeros(4)
        for t in range(N):
            x = np.matrix([[com_err[0,t], com_v[0,t], com_a[0,t], com_j[0,t]]]).T
            u = np.matrix([[com_s[0,t]]])
            state_cost   += dt*(x.T * conf.Q_pos * x)[0,0]
            control_cost += dt*(u.T * conf.R_d4x * u)[0,0]
            for i in range(4): state_cost_component[i] += dt*(x[i] * conf.Q_pos[i,i] * x[i])
        data.cost_state   = np.sqrt(state_cost/T)
        data.cost_control = np.sqrt(control_cost/T)
        data.cost_state_component = np.sqrt(state_cost_component/T)
        
        fric_cone_viol = np.zeros(N)
        for t in range(f.shape[1]):
            tmp = np.max(B_f * f[:,t] - b_f)
            if(tmp>0.0): fric_cone_viol[t] = tmp
        compute_stats_and_add_to_data('fric_cone_viol', fric_cone_viol)
        
        key = res.generate_key(keys, (ctrl, f_dist, zeta, w_d4x))
        res[key] = data
        
        # simulate expected responce
        gains = GainsTsidAdm(optimal_gains[w_d4x])
        K = convert_tsid_adm_gains_to_integrator_gains(gains, K_contact)
        H = A - B*K
        
        x0 = np.matrix([com_err[0,0], com_v[0,0], com_a[0,0], com_j[0,0]]).T

#        np.set_printoptions(precision=3, linewidth=200, suppress=False)
        print 'x0', x0.T
#        print 'q0', lgr.get_vector('simu_q', 8)[:,N_DISTURB_BEGIN].T
#        print 'v0', lgr.get_vector('simu_v', 7)[:,N_DISTURB_BEGIN].T
#        np.set_printoptions(precision=1, linewidth=200, suppress=True)
        
        def compute_expected_costs(x, u):
            expected_state_cost, expected_control_cost, expected_state_cost_component = 0.0, 0.0, np.zeros(4)
            for t in range(N-1):
                expected_state_cost   += dt*(x[:,t].T * conf.Q_pos * x[:,t])[0,0]
                expected_control_cost += dt*(u[:,t].T * conf.R_d4x * u[:,t])[0,0]
                for i in range(4): expected_state_cost_component[i] += dt*(x[i,t] * conf.Q_pos[i,i] * x[i,t])
            return np.sqrt(expected_state_cost/T), np.sqrt(expected_control_cost/T), np.sqrt(expected_state_cost_component/T)
            
        (x,u) = simulate_LDS(A, B, K, x0, dt, N)
        data.expected_state_cost, data.expected_control_cost, data.expected_state_cost_component = compute_expected_costs(x, u)
        
        (x2,u2) = simulate_LDS(A, B, K, conf.x0, dt, N)
        data.expected_state_cost2, data.expected_control_cost2, data.expected_state_cost_component2 = compute_expected_costs(x2, u2)
        
        print 'Real state cost:        %f'%(data.cost_state)
        print 'Expected state cost:    %f'%(data.expected_state_cost)
        print 'Real ctrl cost:         %f'%(data.cost_control)
        print 'Expected ctrl cost:     %f'%(data.expected_control_cost)
        for i in range(4):
            print 'Real state component %d cost:        %f'%(i, data.cost_state_component[i])
            print 'Expected state component %d cost:    %f'%(i, data.expected_state_cost_component[i])
        
        if(plut.SAVE_FIGURES or conf.do_plots):
            # plot real CoM trajectory VS expected CoM trajectory        
            fi, ax = plt.subplots(5, 1, sharex=True);
            i = 0
            ax[i].plot(time, x[0,:].A1, label='expected')
            ax[i].plot(time, (com_p-com_ref)[0,:].A1, '--', label='real')
            ax[i].set_title(r'CoM Pos Y, $w_{d4x}$='+str(w_d4x))
            ax[i].legend()
            i += 1
            ax[i].plot(time, x[ny,:].A1, label='expected')
            ax[i].plot(time, com_v[0,:].A1, '--', label='real')
            ax[i].set_title('CoM Vel Y')
            i += 1
            ax[i].plot(time, x[2*ny,:].A1, label='expected')
            ax[i].plot(time, com_a[0,:].A1, '--', label='real')
            ax[i].set_title('Com Acc Y')
            i += 1
            ax[i].plot(time, x[3*ny,:].A1, label='expected')
            ax[i].plot(time, com_j[0,:].A1, '--', label='real')
            ax[i].set_title('Com Jerk Y')
            i += 1
            ax[i].plot(time[:-1], u[0,:].A1, label='expected')
            ax[i].plot(time, com_s[0,:].A1, '--', label='real')
            ax[i].set_title('Com Snap Y')
            plut.saveFigure('exp_vs_real_com_Y_'+ctrl+'_w_d4x_'+str(w_d4x))
            
#            plt.show()
            
#            fi, ax = plt.subplots(ny, 1, sharex=True);
#            for i in range(ny):
#                ax[i].plot(time, x[i,:].A1, label='expected')
#                if(i<2): ax[i].plot(time[:-N_DISTURB_BEGIN], (com_p-com_ref)[i,N_DISTURB_BEGIN:].A1, '--', label='real')
#                ax[i].set_title('CoM '+str(i)+r', $w_{\ddot{f}}$='+str(w_d4x))
#            ax[0].legend()
#            plut.saveFigure('exp_vs_real_com_'+ctrl+'_w_d4x_'+str(w_d4x))

    
    if(SAVE_DATA):
        print "Save results in", DATA_DIR + OUTPUT_DATA_FILE_NAME+'.pkl'
        with open(DATA_DIR + OUTPUT_DATA_FILE_NAME+'.pkl', 'wb') as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)


keys_sorted = optimal_gains.keys()
keys_sorted.sort()
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure()
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

if(conf.do_plots):
    plt.show()
