# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:57:32 2018

@author: adelprete
"""

from numpy import matlib
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from simple_biped.hrp2_reduced import Hrp2Reduced
from simple_biped.utils.logger import RaiLogger
from simple_biped.utils.utils_thomas import finite_diff, compute_stats
from simple_biped.utils.tsid_utils import createContactForceInequalities
import simple_biped.utils.plot_utils as plut
from simple_biped.utils.regex_dict import RegexDict
from simple_biped.robot_model_path import pkg, urdf 
import os
import itertools
import pickle

class Empty:
    pass    
    
np.set_printoptions(precision=1, linewidth=200, suppress=True)


# User parameters
keys                    = ['ctrl', 'fDist', 'com_sin_amp', 'zeta', 'k']
controllers             = ['tsid_rigid', 'tsid_flex', 'tsid_adm', 'tsid_mistry', 'adm_ctrl']
f_dists                 = [400.]
com_sin_amps            = [0.0]
#zetas                   = [0.1, 0.2, 0.3, 0.5]
#ks                      = [1., 0.1]
zetas                   = [0.1, 0.2, 0.4, 0.8]
ks                      = [0.1, 0.2, 0.4, 1.0, 2.0]
#T                       = 2.0
DATA_DIR                = os.getcwd()+'/../data/data_comparison_paper_v4/'
#DATA_DIR               = str(os.path.dirname(os.path.abspath(__file__)))+'../data/data_comparison_paper_v3/'
DATA_FILE_NAME          = 'logger_data.npz'
OUTPUT_DATA_FILE_NAME   = 'summary_data'
SAVE_DATA               = 1
LOAD_DATA               = 1
plut.SAVE_FIGURES       = 1
SHOW_FIGURES            = 0
dt                      = 1e-3
mu                      = 0.3

plut.FIGURE_PATH        = DATA_DIR
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
    
    for (ctrl, f_dist, com_sin_amp, zeta, k) in itertools.product(controllers, f_dists, com_sin_amps, zetas, ks):
        TEST_DESCR_STR = ctrl + '_zeta_'+str(zeta) + '_k_'+str(k)
        if(com_sin_amp!=0.0): TEST_DESCR_STR += '_comSinAmp_'+str(com_sin_amp)
        if(f_dist!=0.0):      TEST_DESCR_STR += '_fDist_'+str(f_dist)
        INPUT_FILE = DATA_DIR+TEST_DESCR_STR+'/'+DATA_FILE_NAME
        
        print '\n'+"".center(120, '#')
        print ("Gonna read %s"%(TEST_DESCR_STR)).center(120)
        print "".center(120, '#')
            
        # SETUP LOGGER
        lgr = RaiLogger()
        try:
            lgr.load(INPUT_FILE)
        except:
            print "Could not read file", INPUT_FILE
            continue
        
        T = len(lgr.get_streams('simu_q_0'))
        time = np.arange(0.0, dt*T, dt)
    
        robot = Hrp2Reduced(urdf,[pkg],loadModel=True,useViewer=0)
    
        nq = robot.model.nq
        nv = robot.model.nv
        na = robot.model.nv-3
        nf = 4
        q   = lgr.get_vector('simu_q', nq)
        v   = lgr.get_vector('simu_v', nv)
        dv  = lgr.get_vector('simu_dv', nv)
        com_p = lgr.get_vector('simu_com_p', 2)
        com_v = lgr.get_vector('simu_com_v', 2)
        com_a = lgr.get_vector('simu_com_a', 2)
        com_j = lgr.get_vector('simu_com_j', 2)
        com_s = finite_diff(com_j, dt, False)
        com_ref = lgr.get_vector('tsid_comref', 2)
        tau = lgr.get_vector('tsid_tau', na)
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
            
        compute_stats_and_add_to_data('com_pos', com_p-com_ref)
        compute_stats_and_add_to_data('com_vel', com_v)
        compute_stats_and_add_to_data('com_acc', com_a)
        compute_stats_and_add_to_data('com_jerk', com_j)
        compute_stats_and_add_to_data('com_snap', com_s)
        compute_stats_and_add_to_data('df',      df)
        compute_stats_and_add_to_data('ddf',     ddf)
        print "CoM pos tracking RMSE:    %.1f mm"%(1e3*data.com_pos_rmse)
        print "CoM jerk RMSE:            %.1f m/s^3"%(data.com_jerk_rmse)
        print "CoM snap RMSE:            %.1f m/s^4"%(data.com_snap_rmse)
        print "Force vel RMS:            %.1f N/s"%(data.df_rmse)
        print "Force acc RMS:            %.1f N/s^2"%(data.ddf_rmse)
        
        fric_cone_viol = np.zeros(T)
        for t in range(f.shape[1]):
            tmp = np.max(B_f * f[:,t] - b_f)
            if(tmp>0.0): fric_cone_viol[t] = tmp
        compute_stats_and_add_to_data('fric_cone_viol', fric_cone_viol)
        
        key = res.generate_key(keys, (ctrl, f_dist, com_sin_amp, zeta, k))
        res[key] = data
    
    if(SAVE_DATA):
        print "Save results in", DATA_DIR + OUTPUT_DATA_FILE_NAME+'.pkl'
        with open(DATA_DIR + OUTPUT_DATA_FILE_NAME+'.pkl', 'wb') as f:
            pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
        

def plot_from_multikey_dict(d, keys, x_var, y_var, fixed_params, variab_param, yscale=None):
#    print '\n', (' Elements matching pattern '+str(fixed_params)+' ').center(100,'#')
    matching_keys = d.get_matching_keys(keys, fixed_params)
    x, y = {}, {}
    for mk in matching_keys:
        z_i = d.extract_key_value(mk, variab_param)
        if z_i not in x: x[z_i], y[z_i] = [], []    
        data = d[mk]
        x[z_i] += [float(d.extract_key_value(mk, x_var))]
        y[z_i] += [data.__dict__[y_var]]
#        print (' '+mk+' ').center(80,'-'), "\n%s %.3f"%(y_var.ljust(20), y[z_i][-1])
    
    plt.figure()
    for key in x.keys():
        xy = np.array(zip(x[key], y[key]), dtype=[('x', float), ('y', float)])
        xy_sorted = np.sort(xy, order='x')
        x_sorted = [xi for (xi,yi) in xy_sorted]
        y_sorted = [yi for (xi,yi) in xy_sorted]
        plt.plot(x_sorted, y_sorted, '-*', label=key)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.legend()
    if yscale is not None: plt.yscale(yscale)
    fig_name = y_var+'_VS_'+x_var
    for (k,v) in zip(keys, fixed_params):
        if(v is not None):
            fig_name += '__'+k+'='+str(v)
    plut.saveFigure(fig_name)
    return (x,y)

err_kinds = ['mean_err', 'rmse', 'max_err']
for ek in err_kinds:
    # keys =    ['ctrl', 'fDist', 'com_sin_amp', 'zeta', 'k']
    yscale = 'log'
    fDist, k = 400.0, 1.0
    plot_from_multikey_dict(res, keys, 'zeta', 'com_pos_'+ek,       fixed_params=[None, fDist, 0.0, None, k], variab_param='ctrl', yscale=yscale)
    plot_from_multikey_dict(res, keys, 'zeta', 'com_jerk_'+ek,      fixed_params=[None, fDist, 0.0, None, k], variab_param='ctrl', yscale=yscale)
    plot_from_multikey_dict(res, keys, 'zeta', 'com_snap_'+ek,      fixed_params=[None, fDist, 0.0, None, k], variab_param='ctrl', yscale=yscale)
    #plot_from_multikey_dict(res, keys, 'zeta', 'df_norm',           fixed_params=[None, fDist, 0.0, None, k], variab_param='ctrl')
    plot_from_multikey_dict(res, keys, 'zeta', 'ddf_'+ek,           fixed_params=[None, fDist, 0.0, None, k], variab_param='ctrl', yscale=yscale)
    plot_from_multikey_dict(res, keys, 'zeta', 'fric_cone_viol_'+ek, fixed_params=[None, fDist, 0.0, None, k], variab_param='ctrl', yscale=yscale)
    
    #fDist, k = 400.0, 0.1
    #plot_from_multikey_dict(res, keys, 'zeta', 'com_pos_'+ek,       fixed_params=[None, fDist, 0.0, None, k], variab_param='ctrl', yscale=yscale)
    #plot_from_multikey_dict(res, keys, 'zeta', 'com_jerk_'+ek,      fixed_params=[None, fDist, 0.0, None, k], variab_param='ctrl', yscale=yscale)
    #plot_from_multikey_dict(res, keys, 'zeta', 'com_snap_'+ek,      fixed_params=[None, fDist, 0.0, None, k], variab_param='ctrl', yscale=yscale)
    ##plot_from_multikey_dict(res, keys, 'zeta', 'df_'+ek,           fixed_params=[None, fDist, 0.0, None, k], variab_param='ctrl', yscale=yscale)
    #plot_from_multikey_dict(res, keys, 'zeta', 'ddf_'+ek,           fixed_params=[None, fDist, 0.0, None, k], variab_param='ctrl', yscale=yscale)
    #plot_from_multikey_dict(res, keys, 'zeta', 'fric_cone_viol_'+ek, fixed_params=[None, fDist, 0.0, None, k], variab_param='ctrl', yscale=yscale)
    
    zeta = 0.2
    plot_from_multikey_dict(res, keys, 'k',    'com_pos_'+ek,       fixed_params=[None, fDist, 0.0, zeta, None], variab_param='ctrl', yscale=yscale)
    plot_from_multikey_dict(res, keys, 'k',    'com_jerk_'+ek,      fixed_params=[None, fDist, 0.0, zeta, None], variab_param='ctrl', yscale=yscale)
    plot_from_multikey_dict(res, keys, 'k',    'com_snap_'+ek,      fixed_params=[None, fDist, 0.0, zeta, None], variab_param='ctrl', yscale=yscale)
    plot_from_multikey_dict(res, keys, 'k',    'ddf_'+ek,           fixed_params=[None, fDist, 0.0, zeta, None], variab_param='ctrl', yscale=yscale)
    plot_from_multikey_dict(res, keys, 'k',    'fric_cone_viol_'+ek, fixed_params=[None, fDist, 0.0, zeta, None], variab_param='ctrl', yscale=yscale)

    if(SHOW_FIGURES):
        plt.show()
