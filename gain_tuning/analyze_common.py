# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:57:32 2018

Analyze the performance of a controller with different sets of gains.

@author: adelprete
"""

from numpy import matlib
import numpy as np
import matplotlib.pyplot as plt

from simple_biped.utils.logger import RaiLogger
from simple_biped.utils.utils_thomas import compute_stats
from simple_biped.utils.tsid_utils import createContactForceInequalities
from simple_biped.utils.regex_dict import RegexDict
from simple_biped.utils.LDS_utils import simulate_LDS
import simple_biped.utils.plot_utils as plut
from simple_biped.utils.utils_thomas import finite_diff

import itertools
import pickle

np.set_printoptions(precision=1, linewidth=200, suppress=True)

class Empty:
    pass    

def compute_stats_and_add_to_data(data, name, x):
    mean_err, rmse, max_err = compute_stats(x)
    data.__dict__[name+'_mean_err']  = mean_err
    data.__dict__[name+'_rmse']      = rmse
    data.__dict__[name+'_max_err']   = max_err
    data.__dict__[name+'_mse']       = rmse**2
    

def compute_cost_function_matrix(n, w_x, w_dx, w_d2x, w_d3x, w_d4x):
    ''' Convert the cost function of the CoM state in the cost function
        of the state of the linearized closed-loop dynamic
    '''
    Q_diag  = np.matrix(n*[w_x] + n*[w_dx] + n*[w_d2x] + n*[w_d3x] + n*[w_d4x])
    return matlib.diagflat(Q_diag) 
    
def compute_costs(x, u, dt, P, Q_x, Q_u):
    N = x.shape[1]
    T = N*dt
    xu_proj = matlib.empty((P.shape[0], N))  # com state
    state_cost, control_cost = 0.0, 0.0
    for t in range(N):
        if(t<N-1):
            xup = P * np.vstack([x[:,t], u[:,t]])
        else:
            xup = P * np.vstack([x[:,t], matlib.zeros_like(u[:,0])])
        xu_proj[:,t] = xup
        state_cost   += dt*(xup.T * Q_x * xup)[0,0]
        control_cost += dt*(xup.T * Q_u * xup)[0,0]
    return xu_proj, np.sqrt(state_cost/T), np.sqrt(control_cost/T)
    

def plot_real_vs_expected_com_state(time, com_real, com_expected, nc, w_d4x, ctrl, name=''):
    fi, ax = plt.subplots(5, 1, sharex=True);
    i = 0
    ax[i].plot(time, com_expected[0,:].A1, label='expected')
    ax[i].plot(time, com_real[0,:].A1, '--', label='real')
    ax[i].set_title(r'CoM Pos Y, log($w_{u}$)=%.2f'%(np.log10(w_d4x)))
    ax[i].legend()
    i += 1
    ax[i].plot(time, com_expected[nc,:].A1, label='expected')
    ax[i].plot(time, com_real[nc,:].A1, '--', label='real')
    ax[i].set_title('CoM Vel Y')
    i += 1
    ax[i].plot(time, com_expected[2*nc,:].A1, label='expected')
    ax[i].plot(time, com_real[2*nc,:].A1, '--', label='real')
    ax[i].set_title('Com Acc Y')
    i += 1
    ax[i].plot(time, com_expected[3*nc,:].A1, label='expected')
    ax[i].plot(time, com_real[3*nc,:].A1, '--', label='real')
    ax[i].set_title('Com Jerk Y')
    i += 1
    ax[i].plot(time, com_expected[4*nc,:].A1, label='expected')
    ax[i].plot(time, com_real[4*nc,:].A1, '--', label='real')
    ax[i].set_title('Com Snap Y')
    plut.saveFigure('exp_vs_real_com_Y_'+ctrl+'_w_u_%.2f'%(np.log10(w_d4x))+name)
    

def analyze_results(conf, compute_system_matrices, P):
    '''
    @param compute_system_matrices Function that takes ctrl gains as input and computes matrices A, B, K such that dynamic is dx = A*x+B*u, and control is u=-K*x
    @param P Matrix projecting the linear system state+ctrl into the CoM state+ctrl [c, dc, d2c, d3c, d4c]
    '''
    # User parameters
    keys            = conf.keys 
    controllers     = conf.controllers 
    f_dists         = conf.f_dists
    zetas           = conf.zetas
    T               = conf.T_simu
    w_d4x_list      = conf.w_d4x_list
    
    DATA_DIR                = conf.DATA_DIR + conf.TESTS_DIR_NAME + conf.controllers[0] +'/'
    GAINS_DIR               = conf.DATA_DIR + conf.GAINS_DIR_NAME
    dt                      = conf.dt_simu    
    plut.SAVE_FIGURES       = conf.SAVE_FIGURES
    SHOW_FIGURES            = conf.do_plots
    plut.FIGURE_PATH        = DATA_DIR
    
    # try to load file containing optimal gains
    f = open(GAINS_DIR+conf.GAINS_FILE_NAME+'.pkl', 'rb')
    optimal_gains = pickle.load(f)
    f.close()
    
    N = int(T/dt)       # number of time steps
    N0 = 0              # first time step to consider
    nc = conf.nc        # CoM state size
    nf = conf.nf        # contact force size
        
    if(conf.LOAD_DATA):
        # try to load data from pickle file
        try:
            f = open(DATA_DIR+conf.OUTPUT_DATA_FILE_NAME+'.pkl', 'rb');
            res = pickle.load(f); 
            f.close();
        except:
            print "Impossible to open file", DATA_DIR+conf.OUTPUT_DATA_FILE_NAME+'.pkl';
            conf.LOAD_DATA = 0
    
    if(not conf.LOAD_DATA):
        res = RegexDict()
    
        # compute friction cone inequalities
        (B, b) = createContactForceInequalities(conf.mu) # B_f * f <= b_f
        k = B.shape[0]
        B_f = matlib.zeros((2*k,4));
        b_f = matlib.zeros(2*k).T;
        B_f[:k,:2] = B_f[k:,2:] = B
        b_f[:k,0] = b_f[k:,0] = b
                
        # compute cost function matrices
        Q_x, Q_u = 2*[None], 2*[None]
        Q_x[0] = compute_cost_function_matrix(nc, conf.w_x, conf.w_dx, conf.w_d2x, conf.w_d3x, 0.0)
        Q_u[0] = compute_cost_function_matrix(nc, 0.0, 0.0, 0.0, 0.0, 1.0)
        Q_x[1] = compute_cost_function_matrix(nc, conf.w_x, conf.w_dx, conf.w_d2x, conf.w_d3x, 0.0)
        Q_u[1] = compute_cost_function_matrix(nc, 0.0, 0.0, 0.0, 1.0, 0.0)

        P_pinv = np.linalg.pinv(P)
        
        for (ctrl, f_dist, zeta, w_d4x) in itertools.product(controllers, f_dists, zetas, w_d4x_list):
    
            test_name = conf.get_test_name(ctrl, zeta, f_dist, w_d4x)
            INPUT_FILE = DATA_DIR + test_name + '/' + conf.DATA_FILE_NAME
            
            lw = 60
            print '\n'+"".center(lw, '#')
            print ("Gonna read %s"%(test_name)).center(lw)
            print "".center(lw, '#')
                
            # SETUP LOGGER
            data = Empty()
            lgr = RaiLogger()
            INPUT_FILE_FOUND = False
            try:
                lgr.load(INPUT_FILE)
                INPUT_FILE_FOUND = True
            except Exception as e:
                print e
                
            if INPUT_FILE_FOUND:
                com_p = lgr.get_vector('simu_com_p', nc)[:,N0:]
                com_ref = lgr.get_vector('tsid_comref', nc)[:,N0:]
                com_err = com_p-com_ref
                com_v = lgr.get_vector('simu_com_v', nc)[:,N0:]
                com_a = lgr.get_vector('simu_com_a', nc)[:,N0:]
                com_j = lgr.get_vector('simu_com_j', nc)[:,N0:]
                com_s = np.matrix(finite_diff(com_j, dt, False))
                lkf = lgr.get_vector('simu_lkf', nf/nc)[:,N0:]
                rkf = lgr.get_vector('simu_rkf', nf/nc)[:,N0:]
                f   = np.vstack((lkf,rkf))
                
                # compute state cost and control cost for real system
                N = com_p.shape[1]
                time = np.arange(N*dt, step=dt)
                x = matlib.empty((4*nc, N))
                for t in range(N):
                    x[:,t] = np.vstack([com_err[:,t], com_v[:,t], com_a[:,t], com_j[:,t]])
                    
                com_real, data.cost_state, data.cost_control      = compute_costs(x, com_s, dt, matlib.eye(5*nc), Q_x[0], Q_u[0])
                com_real, data.cost_state, data.cost_control_jerk = compute_costs(x, com_s, dt, matlib.eye(5*nc), Q_x[1], Q_u[1])
                data.jerk_max = np.max(np.abs(com_j))
                
                # compute violations of friction cone constraints
                fric_cone_viol = np.zeros(N)
                fz_min = 1.0
                for t in range(f.shape[1]):
                    fz = np.array([f[1,t], f[1,t], f[3,t], f[3,t]])
                    if(f[1,t]==0.0): fz[0] = fz[1] = fz_min
                    if(f[3,t]==0.0): fz[2] = fz[3] = fz_min
                    tmp = np.max(np.divide(B_f * f[:,t] - b_f, fz))
                    if(tmp>0.0): fric_cone_viol[t] = tmp
                compute_stats_and_add_to_data(data, 'fric_cone_viol', fric_cone_viol)
                
                x0_com = np.vstack((com_err[:,0], com_v[:,0], com_a[:,0], com_j[:,0], 0.0*com_s[:,0]))
                xu0 = P_pinv * x0_com
                
            else:
                data.cost_state   = np.nan
                data.cost_control = np.nan
                data.cost_control_jerk = np.nan
                data.jerk_max = np.nan
                data.fric_cone_viol_max_err = np.nan
                time = np.arange(N*dt, step=dt)
                xu0 = conf.x0
                com_real = np.nan*matlib.zeros((5*nc, N))
            
            key = res.generate_key(keys, (ctrl, f_dist, zeta, w_d4x))
            res[key] = data
            
            # simulate linear system to compute expected cost
            A, B, K = compute_system_matrices(optimal_gains[w_d4x]);                    
            x0 = xu0[:A.shape[0],0]
            
            (x,u) = simulate_LDS(A, B, K, x0, dt, N)
            xu_proj, data.expected_state_cost, data.expected_control_cost      = compute_costs(x, u, dt, P, Q_x[0], Q_u[0])
            xu_proj, data.expected_state_cost, data.expected_control_cost_jerk = compute_costs(x, u, dt, P, Q_x[1], Q_u[1])
            data.expected_jerk_max = np.max(np.abs(xu_proj[:,3*nc:4*nc]))
            
#            (x2,u2) = simulate_LDS(A, B, K, conf.x0, dt, N)
#            xu2_proj, data.expected_state_cost2, data.expected_control_cost2, data.expected_state_cost_component2 = \
#                compute_costs(x2, u2, dt, P, Q_x, Q_u, Q_xi)
            
            print 'Real state cost:        %f'%(data.cost_state)
            print 'Expected state cost:    %f'%(data.expected_state_cost)
            print 'Real ctrl cost:         %f'%(data.cost_control)
            print 'Expected ctrl cost:     %f'%(data.expected_control_cost)
            print 'Real max jerk:          %f'%(data.jerk_max)
#            print 'Friction cone violation %f'%(data.fric_cone_viol_max_err)
            print ''
                
            if(plut.SAVE_FIGURES or conf.do_plots):
                # plot real CoM trajectory VS expected CoM trajectory
                plot_real_vs_expected_com_state(time, com_real, xu_proj, nc, w_d4x, ctrl)
#                plt.figure()
#                plt.plot(time, com_j[0,:].A1)
#                print 'max jerk:', np.max(np.abs(com_j[0,:])), np.max(np.abs(com_j))
#                plt.show()
#                plot_real_vs_expected_com_state(time, com_real, xu2_proj, nc, w_d4x, ctrl, 'fixed_x0')
                
        if(conf.SAVE_DATA):
            print "Save results in", DATA_DIR + conf.OUTPUT_DATA_FILE_NAME+'.pkl'
            with open(DATA_DIR + conf.OUTPUT_DATA_FILE_NAME+'.pkl', 'wb') as f:
                pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    
    # plot expected costs VS real costs
    keys_sorted = optimal_gains.keys()
    keys_sorted.sort()
        
    plt.figure()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for (w_d4x, color) in zip(keys_sorted, colors):
        tmp = res.get_matching(keys, [None, None, None, w_d4x]).next()
        label = r'Expected, log($w_{u}$)=%.2f'%(np.log10(w_d4x))
        plt.plot(tmp.expected_state_cost, tmp.expected_control_cost, ' *', color=color, markersize=30, label=label)
        plt.plot(tmp.cost_state, tmp.cost_control, ' o', color=color, markersize=30) #, label='real w_d4x='+str(w_d4x))
    plt.legend()
    plt.grid(True);
    plt.xlabel(r'State cost')
    plt.ylabel(r'Control (snap) cost')
    plt.xscale('log')
    plt.yscale('log')
    plut.saveFigure('roc_performance')

    plt.figure()    
    for (w_d4x, color) in zip(keys_sorted, colors):
        tmp = res.get_matching(keys, [None, None, None, w_d4x]).next()
        label = r'Expected, log($w_{u}$)=%.2f'%(np.log10(w_d4x))
        plt.plot(tmp.expected_state_cost, tmp.expected_control_cost_jerk, ' *', color=color, markersize=30, label=label)
        plt.plot(tmp.cost_state, tmp.cost_control_jerk, ' o', color=color, markersize=30) #, label='real w_d4x='+str(w_d4x))
    plt.legend()
    plt.grid(True);
    plt.xlabel(r'State cost')
    plt.ylabel(r'Control (jerk) cost')
    plt.xscale('log')
    plt.yscale('log')
    plut.saveFigure('roc_performance_jerk')
    
    plt.figure()    
    for (w_d4x, color) in zip(keys_sorted, colors):
        tmp = res.get_matching(keys, [None, None, None, w_d4x]).next()
        label = r'Expected, log($w_{u}$)=%.2f'%(np.log10(w_d4x))
        plt.plot(tmp.expected_state_cost, tmp.expected_control_cost_jerk, ' *', color=color, markersize=30, label=label)
        plt.plot(tmp.cost_state, tmp.jerk_max, ' o', color=color, markersize=30) #, label='real w_d4x='+str(w_d4x))
    plt.legend()
    plt.grid(True);
    plt.xlabel(r'State cost')
    plt.ylabel(r'Control (jerk max) cost')
    plt.xscale('log')
    plt.yscale('log')
    plut.saveFigure('roc_performance_jerk_max')
    
#    plt.figure()
#    for (w_d4x, color) in zip(keys_sorted, colors):
#        tmp = res.get_matching(keys, [None, None, None, w_d4x]).next()
#        if(tmp.fric_cone_viol_max_err==0.0):
#            tmp.fric_cone_viol_max_err = 1e-5
#        label = r'log($w_{u}$)=%.2f'%(np.log10(w_d4x))
#        plt.plot(tmp.cost_state, tmp.fric_cone_viol_max_err, ' o', color=color, markersize=30, label=label)
#    plt.legend()
#    plt.grid(True);
#    plt.xlabel(r'State cost')
#    plt.ylabel(r'Friction violation')
#    plt.xscale('log')
#    plt.yscale('log')
#    plut.saveFigure('roc_friction_violation')
    
    if(SHOW_FIGURES):
        plt.show()


def plot_performance(name, res_list, conf_list, marker_list, xlabel, ylabel,
                     x_perf, y_perf, 
                     x_min, y_min,
                     x_max, y_max, 
                     x_max_replacement, y_max_replacement):
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    points = []
    for (conf, res, mark) in zip(conf_list, res_list, marker_list):
        w_u_list   = conf.w_d4x_list
        label_done = False
        for (i, (w_u, color)) in enumerate(zip(w_u_list, colors)):
            tmp = res.get_matching(conf.keys, [None, None, None, w_u]).next()
            p = Empty()
            p.x = tmp.__dict__[x_perf]
            p.y = tmp.__dict__[y_perf]
            p.w_u = w_u
            p.conf = conf
            p.mark = mark
            p.color = color
            p.lbl = ''
            
            if np.isfinite(p.x) and np.isfinite(p.y):
                points += [p]
            else:
                continue
            
            if p.x>x_max:  p.x = x_max_replacement
            if p.y>y_max:  p.y = y_max_replacement
                
            if not label_done:
                p.lbl = conf.ctrl_long_name
                label_done = True
        
    # find Pareto-optimal points
    optimal_points = []
    suboptimal_points = []
    for p0 in points:
        if np.isnan(p0.x): continue
        p0_optimal = True
        for p1 in points:
            if p1.x<p0.x and p1.y<p0.y:
                p0_optimal = False
                suboptimal_points += [p0]
                break
        if p0_optimal: optimal_points += [p0]
    
    # plot with colors corresponding to values of w_u
    plt.figure()
    for p in points:
        plt.plot(p.x, p.y, ' '+p.mark, color=p.color, markersize=30, alpha = 0.75, label=p.lbl)
    plt.legend();           plt.grid(True);
    plt.xlabel(xlabel);     plt.ylabel(ylabel)
    plt.xscale('log');      plt.yscale('log')
    plt.xlim(x_min, x_max); plt.ylim(y_min, 1.3*y_max)
    plut.saveFigure(name)
    
    # plot with colors corresponding to Pareto optimality
    index = 0
    for p in optimal_points:
        # LOAD DATA FOR PLOT
        test_name = p.conf.get_test_name(p.conf.controllers[0], p.conf.zetas[0], 
                                         p.conf.f_dists[0], p.w_u)
        DATA_DIR = p.conf.DATA_DIR + p.conf.TESTS_DIR_NAME + p.conf.controllers[0] +'/'
        INPUT_FILE = DATA_DIR + test_name + '/' + p.conf.DATA_FILE_NAME
        # SETUP LOGGER
        nc, nf, dt, N0 = conf.nc, conf.nf, conf.dt_simu, 0
        lgr = RaiLogger()
        lgr.load(INPUT_FILE)
        com_p = lgr.get_vector('simu_com_p', nc)[:,N0:]
        com_ref = lgr.get_vector('tsid_comref', nc)[:,N0:]
        com_err = com_p-com_ref
#        com_v = lgr.get_vector('simu_com_v', nc)[:,N0:]
#        com_a = lgr.get_vector('simu_com_a', nc)[:,N0:]
        com_j = lgr.get_vector('simu_com_j', nc)[:,N0:]
#        lkf = lgr.get_vector('simu_lkf', nf/nc)[:,N0:]
#        rkf = lgr.get_vector('simu_rkf', nf/nc)[:,N0:]
#        f   = np.vstack((lkf,rkf))
        # compute state cost and control cost for real system
        N = com_p.shape[1]
        time = np.arange(N*dt, step=dt)
        
        fi, ax = plt.subplots(2, 1, sharex=True);
        i = 0
        ax[i].plot(time, com_err[0,:].A1, '-', label='real')
        ax[i].set_title(p.conf.ctrl_long_name+', State cost: %.1f'%(np.log10(p.x)))
        ax[i].set_ylabel(r'CoM pos Y [$m$]')
        i += 1
        ax[i].plot(time, com_j[0,:].A1, '-', label='real')
        ax[i].set_ylabel(r'CoM jerk Y [$m/s^3$]')
        plut.saveFigure(name+'_pareto_%.2f_%s'%(np.log10(p.x), p.conf.controllers[0]))
        index += 1
        
        
        
    plt.figure()
    for p in optimal_points:
        plt.plot(p.x, p.y, ' '+p.mark, color='r', markersize=30, alpha = 0.75, label=p.lbl)
    for p in suboptimal_points:
        plt.plot(p.x, p.y, ' '+p.mark, color='b', markersize=30, alpha = 0.75, label=p.lbl)
    plt.legend();           plt.grid(True);
    plt.xlabel(xlabel);     plt.ylabel(ylabel)
    plt.xscale('log');      plt.yscale('log')
    plt.xlim(x_min, x_max); plt.ylim(y_min, 1.3*y_max)
    plut.saveFigure(name+'_pareto')
    
    plt.close('all')

def compare_controllers(conf_list, marker_list):   
    MAX_STATE_COST   = 1.0
    MAX_CONTROL_COST = 1e6
    MAX_JERK_COST    = 1e3
    keys                = conf_list[0].keys 
    plut.FIGURE_PATH    = conf_list[0].DATA_DIR + conf_list[0].TESTS_DIR_NAME
    plut.SAVE_FIGURES   = conf_list[0].SAVE_FIGURES
    SHOW_FIGURES        = conf_list[0].do_plots
    
    res_list = len(conf_list)*[None]
    i = 0
    
    for conf in conf_list:
        DATA_FILE_DIR = conf.DATA_DIR + conf.TESTS_DIR_NAME + conf.controllers[0] +'/'
        f = open(DATA_FILE_DIR+conf.OUTPUT_DATA_FILE_NAME+'.pkl', 'rb');
        res_list[i] = pickle.load(f); 
        i += 1
        f.close();
        
    plot_performance('roc_performance_comparison', res_list, conf_list, marker_list, 
                     'State cost', 'Snap cost',
                     'cost_state', 'cost_control', 
                     8e-4, 9.0,
                     MAX_STATE_COST, MAX_CONTROL_COST, 
                     np.nan, MAX_CONTROL_COST)
 
    plot_performance('roc_performance_comparison_jerk', res_list, conf_list, marker_list, 
                     'State cost', 'Jerk cost',
                     'cost_state', 'cost_control_jerk', 
                     8e-4, 1.0,
                     MAX_STATE_COST, MAX_JERK_COST, 
                     np.nan, MAX_JERK_COST)
   
    plot_performance('roc_performance_comparison_jerk_max', res_list, conf_list, marker_list, 
                     'State cost', 'Max Jerk',
                     'cost_state', 'jerk_max', 
                     8e-4, 1.0,
                     MAX_STATE_COST, MAX_JERK_COST, 
                     np.nan, MAX_JERK_COST)
     
    if(SHOW_FIGURES):
        plt.show()


if __name__=='__main__':
    import simple_biped.gain_tuning.conf_adm_ctrl as conf_adm_ctrl
    import simple_biped.gain_tuning.conf_tsid_adm as conf_tsid_adm
    import simple_biped.gain_tuning.conf_tsid_flex_k as conf_tsid_flex_k
    import simple_biped.gain_tuning.conf_tsid_rigid as conf_tsid_rigid
    
    conf_list = [conf_tsid_rigid, conf_adm_ctrl, conf_tsid_adm, conf_tsid_flex_k]
    marker_list = ['s', 'o', '*', 'v']
    
#    conf_list = [conf_tsid_adm, conf_tsid_flex_k]
#    marker_list = ['*', 'x']
    
    compare_controllers(conf_list, marker_list)
    