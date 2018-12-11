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
from simple_biped.gain_tuning.genetic_tuning import GainOptimizeAdmCtrl
from simple_biped.simu import Simu
from simple_biped.admittance_ctrl import GainsAdmCtrl
from simple_biped.hrp2_reduced import Hrp2Reduced
from simple_biped.robot_model_path import pkg, urdf 
import simple_biped.utils.plot_utils as plut

import simple_biped.gain_tuning.adm_ctrl_tuning_conf as conf

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
w_ddf_list      = conf.w_ddf_list

DATA_DIR                = conf.DATA_DIR + conf.TESTS_DIR_NAME
GAINS_DIR               = conf.DATA_DIR + conf.GAINS_DIR_NAME
GAINS_FILE_NAME         = conf.GAINS_FILE_NAME
dt                      = conf.dt_simu
mu                      = conf.mu
T_DISTURB_BEGIN         = conf.T_DISTURB_BEGIN

DATA_FILE_NAME          = 'logger_data.npz'
OUTPUT_DATA_FILE_NAME   = 'summary_data'
SAVE_DATA               = 1
LOAD_DATA               = 1
plut.SAVE_FIGURES       = 1
SHOW_FIGURES            = 1
plut.FIGURE_PATH        = DATA_DIR

# try to load file containing optimal gains
f = open(GAINS_DIR+GAINS_FILE_NAME+'.pkl', 'rb')
optimal_gains = pickle.load(f)
f.close()

N = int(T/dt)
N_DISTURB_BEGIN = int(T_DISTURB_BEGIN/dt)
ny = conf.ny
nf = conf.nf
x0 = conf.x0

# SETUP
robot   = Hrp2Reduced(urdf, [pkg], loadModel=0, useViewer=0)
q       = robot.q0.copy()
v       = matlib.zeros((robot.model.nv,1))
K       = Simu.get_default_contact_stiffness()
initial_gains = GainsAdmCtrl.get_default_gains(K)

Q_pos   = conf.Q_pos
Q_ddf   = conf.Q_ddf 
expected_cost_pos = {}
expected_cost_ddf = {}

gain_optimizer = GainOptimizeAdmCtrl(robot, q, v, K, ny, nf, initial_gains.to_array(), dt, x0, N, Q_pos)


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
    
    for (ctrl, f_dist, zeta, w_ddf) in itertools.product(controllers, f_dists, zetas, w_ddf_list):

        test_name = conf.get_test_name(ctrl, zeta, f_dist, w_ddf)
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
        com_p = lgr.get_vector('simu_com_p', 2)
        com_ref = lgr.get_vector('tsid_comref', 2)
        com_v = lgr.get_vector('simu_com_v', 2)
        lkf = lgr.get_vector('simu_lkf', nf/2)
        rkf = lgr.get_vector('simu_rkf', nf/2)
        f   = np.vstack((lkf,rkf))
        df  = lgr.get_vector('simu_df', nf)
        ddf = lgr.get_vector('simu_ddf', nf)
        
        data = Empty()
        def compute_stats_and_add_to_data(name, x):
            mean_err, rmse, max_err = compute_stats(x)
            data.__dict__[name+'_mean_err']  = mean_err
            data.__dict__[name+'_rmse']      = rmse
            data.__dict__[name+'_max_err']   = max_err
            data.__dict__[name+'_mse']       = rmse**2
            
        compute_stats_and_add_to_data('com_pos', com_p-com_ref)
        compute_stats_and_add_to_data('ddf',     ddf)
        print "CoM pos tracking MSE:    %.1f mm"%(1e3*data.com_pos_mse)
        print "Force acc MS:            %.1f N/s^2"%(data.ddf_mse)
        
        N = len(lgr.get_streams('simu_q_0'))
        fric_cone_viol = np.zeros(N)
        for t in range(f.shape[1]):
            tmp = np.max(B_f * f[:,t] - b_f)
            if(tmp>0.0): fric_cone_viol[t] = tmp
        compute_stats_and_add_to_data('fric_cone_viol', fric_cone_viol)
        
        key = res.generate_key(keys, (ctrl, f_dist, zeta, w_ddf))
        res[key] = data
        
        # simulate expected responce
        H = gain_optimizer.compute_transition_matrix(optimal_gains[w_ddf]);
        x = simulate_ALDS(H, x0, dt, N, 0, 0)
        
        if(plut.SAVE_FIGURES or conf.do_plots):
            # plot real CoM trajectory VS expected CoM trajectory        
            fi, ax = plt.subplots(3, 1, sharex=True);
            ax[0].plot(time, x[0,:].A1, label='expected')
            ax[0].plot(time[:-N_DISTURB_BEGIN], (com_p-com_ref)[0,N_DISTURB_BEGIN:].A1, '--', label='real')
            ax[0].set_title(r'CoM Pos Y, $w_{\ddot{f}}$='+str(w_ddf))
            ax[0].legend()
            ax[1].plot(time, x[ny,:].A1, label='expected')
            ax[1].plot(time[:-N_DISTURB_BEGIN], com_v[0,N_DISTURB_BEGIN:].A1, '--', label='real')
            ax[1].set_title('CoM Vel Y')
            ax[2].plot(time, x[2*ny+2*nf+1,:].A1, label='expected')
            ax[2].plot(time[:-N_DISTURB_BEGIN], ddf[1,N_DISTURB_BEGIN:].A1, '--', label='real')
            ax[2].set_title('ddf left foot Z')
            plut.saveFigure('exp_vs_real_com_pos_vel_ddf_Y_'+ctrl+'_w_ddf_'+str(w_ddf))
            
            fi, ax = plt.subplots(ny, 1, sharex=True);
            for i in range(ny):
                ax[i].plot(time, x[i,:].A1, label='expected')
                if(i<2): ax[i].plot(time[:-N_DISTURB_BEGIN], (com_p-com_ref)[i,N_DISTURB_BEGIN:].A1, '--', label='real')
                ax[i].set_title('CoM '+str(i)+r', $w_{\ddot{f}}$='+str(w_ddf))
            ax[0].legend()
            plut.saveFigure('exp_vs_real_com_'+ctrl+'_w_ddf_'+str(w_ddf))
#            mean_err, rmse, max_err = compute_stats(x[:2,:])
#            print 'expected MSE x %.6f'%(rmse**2)
#            print 'real MSE com   %.6f'%data.com_pos_mse
            
            fi, ax = plt.subplots(nf, 1, sharex=True);
            for i in range(nf):
                ax[i].plot(time, x[2*ny+i,:].A1, label='expected')
                ax[i].plot(time[:-N_DISTURB_BEGIN], f[i,N_DISTURB_BEGIN:].A1, '--', label='real')
                ax[i].set_title('f '+str(i)+r', $w_{\ddot{f}}$='+str(w_ddf))
            ax[0].legend()
            plut.saveFigure('exp_vs_real_f_'+ctrl+'_w_ddf_'+str(w_ddf))
            
            fi, ax = plt.subplots(nf, 1, sharex=True);
            for i in range(nf):
                ax[i].plot(time, x[2*ny+nf+i,:].A1, label='expected')
                ax[i].plot(time[:-N_DISTURB_BEGIN], df[i,N_DISTURB_BEGIN:].A1, '--', label='real')
                ax[i].set_title('df '+str(i)+r', $w_{\ddot{f}}$='+str(w_ddf))
            ax[0].legend()
            plut.saveFigure('exp_vs_real_df_'+ctrl+'_w_ddf_'+str(w_ddf))
            
            fi, ax = plt.subplots(nf, 1, sharex=True);
            for i in range(nf):
                ax[i].plot(time, x[2*ny+2*nf+i,:].A1, label='expected')
                ax[i].plot(time[:-N_DISTURB_BEGIN], ddf[i,N_DISTURB_BEGIN:].A1, '--', label='real')
                ax[i].set_title('ddf '+str(i)+r', $w_{\ddot{f}}$='+str(w_ddf))
            ax[0].legend()
            plut.saveFigure('exp_vs_real_ddf_'+ctrl+'_w_ddf_'+str(w_ddf))
    
    if(SAVE_DATA):
        print "Save results in", DATA_DIR + OUTPUT_DATA_FILE_NAME+'.pkl'
        with open(DATA_DIR + OUTPUT_DATA_FILE_NAME+'.pkl', 'wb') as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)

# computed expected costs based on linear approximation of closed-loop system
keys_sorted = optimal_gains.keys()
keys_sorted.sort()
for w_ddf in keys_sorted:
    gains = optimal_gains[w_ddf]
    normalized_opt_gains = gain_optimizer.normalize_gains_array(gains)
    
    gain_optimizer.set_cost_function_matrix(Q_pos)
    expected_cost_pos[w_ddf]    = gain_optimizer.cost_function(normalized_opt_gains)
    gain_optimizer.set_cost_function_matrix(Q_ddf)
    expected_cost_ddf[w_ddf]    = gain_optimizer.cost_function(normalized_opt_gains)
    
    print("".center(100,'#'))
    print("w_ddf={}".format(w_ddf))
    print("Expected optimal cost pos {}".format(expected_cost_pos[w_ddf]))
    print("Expected optimal cost ddf {}".format(expected_cost_ddf[w_ddf]))

plt.figure()
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for (w_ddf, color) in zip(keys_sorted, colors):
    plt.plot(expected_cost_pos[w_ddf], expected_cost_ddf[w_ddf], ' *', color=color, markersize=30, label=r'Expected, $w_{\ddot{f}}$='+str(w_ddf))
    tmp = res.get_matching(keys, [None, None, None, w_ddf]).next()
    plt.plot(tmp.com_pos_mse, tmp.ddf_mse, ' o', color=color, markersize=30) #, label='real w_ddf='+str(w_ddf))
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True);
plt.xlabel(r'Position tracking cost [m${}^2$]')
plt.ylabel(r'Force acceleration cost [N${}^2$s${}^{-4}$]')
plut.saveFigure('roc_'+controllers[0])

if(conf.do_plots):
    plt.show()
