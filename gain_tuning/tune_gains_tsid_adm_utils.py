# -*- coding: utf-8 -*-
"""
Script to find the gains for an admittance controller.

Created on Thu Jul  5 14:41:32 2018

@author: adelprete
"""

import numpy as np
import numpy.ma as ma
from numpy import matlib
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numpy.linalg import eigvals

from simple_biped.utils.LDS_utils import compute_integrator_gains, compute_integrator_dynamics
from simple_biped.simu import Simu
from simple_biped.tsid_admittance import GainsTsidAdm

np.set_printoptions(precision=2, linewidth=200)
 

def convert_integrator_gains_to_tsid_adm_gains(gains_integrator, K, const = 100.0, dt=None, verbose=False):
    (K1, K2, K3, K4) = gains_integrator.tolist()[0]
    gains = GainsTsidAdm()
    gains.Kf = const*np.linalg.inv(K)           # force feedback gain
    gains.Kd_adm = K4                           # contact point velocity gain
    gains.Kp_adm = K3 / (1+const)               # contact point position gain
    gains.Kd_com = K2 / (const*gains.Kp_adm)    # CoM derivative gain
    gains.Kp_com = K1 / (const*gains.Kp_adm)    # CoM position gain
    
    if(verbose):
        (H, A, B) = compute_integrator_dynamics(gains_4th_order_system)
        print "\nEigenvalues corresponding to desired closed-loop gains of 4-th order system:\n", np.sort_complex(eigvals(H))
        if(dt is not None):
            (H_dt, A, B) = compute_integrator_dynamics(gains_4th_order_system, dt);
            print "\nEigenvalues of discrete-time closed-loop system:\n", np.sort_complex(eigvals(H_dt))
        print "\nCorresponding gains for admittance control:\n", gains.to_string()
    return gains

def convert_tsid_adm_gains_to_integrator_gains(gains_tsid_adm, K, n=1):
    const = gains_tsid_adm.Kf[0,0]*K[0,0]
    In = matlib.eye(n)
    K4 = gains_tsid_adm.Kd_adm * In
    K3 = (1+const)*gains_tsid_adm.Kp_adm * In
    K2 = (const*gains_tsid_adm.Kp_adm)*gains_tsid_adm.Kd_com * In
    K1 = (const*gains_tsid_adm.Kp_adm)*gains_tsid_adm.Kp_com * In
    return np.hstack([K1, K2, K3, K4])

def compute_adm_ctrl_vel_gains(gains_3rd_order_integrator, K, dt, verbose=False):
    ''' This never worked well. Not clear why. '''
    from robot_model_path import pkg, urdf 
    from hrp2_reduced import Hrp2Reduced
    import pinocchio as se3
    from numpy import matlib

    robot = Hrp2Reduced(urdf,[pkg],loadModel=1, useViewer=0)
    q = robot.q0.copy()
    v = matlib.zeros((robot.model.nv,1))
    se3.computeAllTerms(robot.model,robot.data,q,v)
    se3.framesKinematics(robot.model,robot.data,q)
    M = robot.data.M        #(7,7)
    Mj = robot.data.M[3:,3:]        #(7,7)
    Minv = np.linalg.inv(M)
    Jl,Jr = robot.get_Jl_Jr_world(q, False)
    J = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
    Upsilon = J*Minv*J.T
    K_Upsilon = np.diag(K*Upsilon)
    
    print "Upsilon=", np.diag(Upsilon)
    print "K*Upsilon=", K_Upsilon
    
    (K1, K2, K3) = gains_3rd_order_integrator.tolist()
    Kd_j = K3 * np.diag(np.diag(Mj))      # contact point position gain
    Kp_j = (K2 - np.mean(K_Upsilon))*np.diag(np.diag(Mj))    # CoM derivative gain
    Kf   = (K2 - np.diag(K_Upsilon))*np.linalg.inv(K)*K1
    
    if(verbose):
        (H, A, B) = compute_integrator_dynamics(gains_3rd_order_integrator);
        ei = eigvals(H);
        print "\nEigenvalues corresponding to desired closed-loop gains of 3-rd order system:\n", ei
        if(dt is not None):
            (H_dt, A, B) = compute_integrator_dynamics(gains_3rd_order_integrator, dt);
            ei_dt = eigvals(H_dt);
            print "\nEigenvalues of discrete-time closed-loop system:\n", ei_dt
        print "\nCorresponding gains for admittance control:"
        for gain in ['Kf', 'Kp_j', 'Kd_j']:
            print gain+' = np.matrix(np.diag(', np.diag(locals()[gain]), '))'
    return (Kf, Kp_j, Kd_j)


def compute_stats_force_damping_factor(p1_bounds, dp_bounds, dt, N_p, N_dp, log_space):
    if(log_space):
        p1 = np.logspace(np.log10(p1_bounds[0]), np.log10(p1_bounds[1]), N_p)
        dp = np.logspace(np.log10(dp_bounds[0]), np.log10(dp_bounds[1]), N_dp)
    else:
        p1 = np.linspace(p1_bounds[0], p1_bounds[1], N_p)
        dp = np.linspace(dp_bounds[0], dp_bounds[1], N_dp)
    P1,DP = np.meshgrid(p1, dp)
    damp_fact = np.empty((N_p, N_dp))
    for i in range(N_p):
        for j in range(N_dp):
            gains_4th_order_system = compute_integrator_gains(4, P1[i,j], DP[i,j], dt)
            (K1, K2, K3, K4) = gains_4th_order_system.tolist()
            damp_fact[i,j] = 0.5*K4/sqrt(K3)
    
    # mask nan
    damp_fact = ma.masked_invalid(damp_fact)
        
    # plot using different scales
    plot_scales = [('lin', 'lin')]
    for ps in plot_scales:
        fig = plt.figure(figsize = (12,8))
        fig.subplots_adjust(wspace=0.3)
        if(ps[0]=='log'):
            # use log scale for color map
            plt.pcolormesh(-P1, -DP, damp_fact, cmap=plt.cm.get_cmap('Blues'),
                           norm=colors.LogNorm(vmin=damp_fact.min(), vmax=damp_fact.max()))
        else:
            plt.pcolormesh(-P1, -DP, damp_fact, cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        
        # draw a red cross over optimal point
        i = np.unravel_index(np.argmin(damp_fact), damp_fact.shape)
        plt.plot(-P1[i], -DP[i], 'ro', markersize=12)
        
        plt.xlabel('p1')
        plt.ylabel('dp')
        plt.title('Damping Factor')
        if(ps[1]=='log'):
            plt.xscale('log')
            plt.yscale('log')
            
    plt.show()
    return (P1, DP, damp_fact)
    

if __name__=='__main__':
    # INPUT PARAMETERS
    TUNE_TSID_ADM = 1
    TUNE_ADM_CTRL_VEL = 0
    verbose = 1
    K = Simu.get_default_contact_stiffness()
    Ky = K[0,0] #23770.
    Kz = K[1,1] #239018.
    
    if(TUNE_TSID_ADM):
        print "\n\n******************************************************"
        print     "            TUNING TSID-ADMITTANCE GAINS"
        print     "******************************************************"
        dt, p1, dp = None, -5.0, -10.0
        if('des_gains' not in locals()):
            gains_4th_order_system = compute_integrator_gains(4, p1, dp, dt)
            print "Desired gains:", gains_4th_order_system    
        gains = convert_integrator_gains_to_tsid_adm_gains(gains_4th_order_system, K, dt=dt, verbose=verbose)
        #(P1, DP, damp_fact) = compute_stats_force_damping_factor(p1_bounds=[-5, -50], dp_bounds=[-0.1, -10], dt=dt, N_p=20, N_dp=20, log_space=False)
    
    if(TUNE_ADM_CTRL_VEL):
        print "\n\n******************************************************"
        print     "      TUNING ADMITTANCE CONTROL (VELOCITY) GAINS      "
        print     "******************************************************"
        dt, p1, dp, K = None, -40.0, -20.0, np.diagflat([Ky,Kz,Ky,Kz]) 
        gains_3rd_order_integrator = compute_integrator_gains(3, p1, dp, dt)
        print "Desired gains:", gains_3rd_order_integrator    
        compute_adm_ctrl_vel_gains(gains_3rd_order_integrator, K, dt, verbose=1)