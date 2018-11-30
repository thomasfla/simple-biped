# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:57:32 2018

@author: adelprete
"""

import pinocchio as se3
from numpy import matlib
import numpy as np
from numpy.linalg import norm, eigvals
from scipy.linalg import expm
from scipy.optimize import basinhopping

import copy 
from pinocchio.utils import *
from math import pi,sqrt
from simple_biped.hrp2_reduced import Hrp2Reduced
from simple_biped.utils.logger import RaiLogger
import matplotlib.pyplot as plt
import simple_biped.utils.plot_utils as plut
from simple_biped.robot_model_path import pkg, urdf 
import os

class Empty:
    pass

class Gains:
    
    def __init__(self, gain_array=None):
        if gain_array is not None:
            self.from_array(gain_array)
    
    def to_array(self):
        nf = self.Kf.shape[0]
        res = np.zeros(6+nf)
        res[0] = self.Kp_adm
        res[1] = self.Kd_adm
        res[2] = self.Kp_com
        res[3] = self.Kd_com
        res[4] = self.kp_bar
        res[5] = self.kd_bar
        res[6:6+nf] = np.diag(self.Kf)
        return res
        
    def from_array(self, gains):
        self.Kp_adm = gains[0]
        self.Kd_adm = gains[1]
        self.Kp_com = gains[2]
        self.Kd_com = gains[3]
        self.kp_bar = gains[4]
        self.kd_bar = gains[5]
        self.Kf     = np.asmatrix(np.diag(gains[6:]))
        
    def to_string(self):
        for s in ['Kp_adm', 'Kd_adm', 'Kp_com', 'Kd_com', 'kp_bar', 'kd_bar']:
            print s+' = ', self.__dict__[s]
        print 'Kf = 1e-4*np.diag(', [v for v in 1e4*np.diag(self.Kf)], ')'

        

def normalize_gains_array(gains_array, nominal_gains_array):
    return np.divide(gains_array, nominal_gains_array)
    
def denormalize_gains_array(normalized_gains_array, nominal_gains_array):
    return np.multiply(normalized_gains_array, nominal_gains_array)
    
    

def compute_ddf_approximations(robot):
    ''' Compute ddf based on the following dynamics equation:
          ddf_1 = K*(-Upsilon*f - dJ*v + J*M_inv*h - J*Minv*S^T*tau)
    '''
    ddf_list = {}
    for i in range(1,11):
        ddf_list[str(i)]   = matlib.empty((nf,T))*np.nan
    
    dyn_res = matlib.empty((nv,T))*np.nan   # residual of robot dynamics
    S = matlib.zeros((na,nv))
    S[:,nv-na:] = matlib.eye(na)
    dq_cmd_integral = q_cmd[:,0].copy()
    eig_J_Minv_ST_Mj_diag_JSTpinv = np.empty((nf,T))
    
    for t in range(T):
        se3.computeAllTerms(robot.model, robot.data, q[:,t], v[:,t])
        se3.framesKinematics(robot.model, robot.data, q[:,t])
        
        M = robot.data.M        #(7,7)
        h = robot.data.nle
        Jl,Jr = robot.get_Jl_Jr_world(q[:,t], False)
        J = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
        driftLF,driftRF = robot.get_driftLF_driftRF_world(q[:,t], v[:,t], False)
        dJ_v = np.vstack([driftLF.vector[1:3], driftRF.vector[1:3]])
        
        Mj = robot.data.M[3:,3:]        #(7,7)
        Minv = np.linalg.inv(M)    
        Upsilon = J*Minv*J.T
        
        ddf_list['1'][:,t] = K*(-Upsilon*f[:,t] - dJ_v + J*Minv*h - J*Minv*S.T*tau[:,t])
        
        tau_2 = Kp_pos*(q_cmd[:,t]-q[4:,t]) - Kd_pos*v[3:,t]
        ddf_list['2'][:,t] = K*(-Upsilon*f[:,t] - dJ_v + J*Minv*h - J*Minv*S.T*tau_2)
        
        # simplified version of controller (without postural task)
        feet_v_ref = - Kf*(f_des[:,t] - f[:,t])
        JSTpinv = np.linalg.pinv(J*S.T)
        dq_cmd = JSTpinv * feet_v_ref
        dq_cmd_integral += dt*dq_cmd
        if(t==0): dq_cmd_integral = q_cmd[:,0]
        tau_3 = Kp_pos*(dq_cmd_integral-q[4:,t]) - Kd_pos*v[3:,t]
        ddf_list['3'][:,t] = K*(-Upsilon*f[:,t] - dJ_v + J*Minv*h - J*Minv*S.T*tau_3)
        
        # compute eigenvalues of J*Minv*S.T*Mj_diag*JSTpinv to check its positive-definiteness
        A = J*Minv*S.T*Mj_diag*JSTpinv
        eig_J_Minv_ST_Mj_diag_JSTpinv[:,t] = eigvals(A)
        B = Minv*S.T*Mj_diag*S
        B_bar = B - matlib.eye(B.shape[0])
        
        # move the pseudoinverse out of the integral
        if(t==0): 
            feet_v_ref_integral_0 = J*S.T*q_cmd[:,0]
            e_f_integral_0 = Kf_inv*feet_v_ref_integral_0
            feet_v_ref_integral = feet_v_ref_integral_0
        else:     
            feet_v_ref_integral += dt*feet_v_ref
        tau_4 = Kp_pos*(JSTpinv*feet_v_ref_integral - q[4:,t]) - Kd_pos*v[3:,t]
        ddf_list['4'][:,t] = K*(-Upsilon*f[:,t] - dJ_v + J*Minv*h - J*Minv*S.T*tau_4)
        
        # equivalent reformulation
        tau_5 = kp_bar*A*feet_v_ref_integral - kp_bar*J*B*q[1:,t] + kd_bar*K_inv*df[:,t] - kd_bar*J*B_bar*v[:,t]
        ddf_list['5'][:,t] = K*(-Upsilon*f[:,t] - dJ_v + J*Minv*h - tau_5)
        
        # remove dJ_v
        e_f_integral = Kf_inv*feet_v_ref_integral
        tau_6 = kp_bar*A*Kf*e_f_integral - kp_bar*J*B*q[1:,t] + kd_bar*K_inv*df[:,t] - kd_bar*J*B_bar*v[:,t]
        ddf_list['6'][:,t] = K*(-Upsilon*f[:,t] + J*Minv*h - tau_6)
        
        # equivalent reformulation 
        ddf_list['7'][:,t] = -kd_bar*df[:,t] -K*Upsilon*f[:,t] -kp_bar*K*A*Kf*e_f_integral +K*J*Minv*h  +kp_bar*K*J*B*q[1:,t] +kd_bar*K*J*B_bar*v[:,t]
        
        # neglect B_bar
        ddf_list['8'][:,t] = -kd_bar*df[:,t] -K*Upsilon*f[:,t] -kp_bar*K*A*Kf*e_f_integral +K*J*Minv*h  +kp_bar*K*J*B*q[1:,t]
        
        # neglect J*B*q
        ddf_list['9'][:,t] = -kd_bar*df[:,t] -K*Upsilon*f[:,t] -kp_bar*K*A*Kf*e_f_integral +K*J*Minv*h                         +kd_bar*K*J*B_bar*v[:,t]
        
        # neglect K*J*Minv*h and assume zero integral at start
        ddf_list['10'][:,t] = -kd_bar*df[:,t] -K*Upsilon*f[:,t] -kp_bar*K*A*Kf*(e_f_integral-e_f_integral_0) +kp_bar*K*J*B*q[1:,t] +kd_bar*K*J*B_bar*v[:,t]
        
        dyn_res[:,t] = M*dv[:,t] + h - J.T*f[:,t] - S.T*tau[:,t]
        
        if(t*dt>=T_DISTURB_BEGIN and t*dt<=T_DISTURB_END):
            ddf[:,t] = np.nan
            dyn_res[:,t] = np.nan
            for key in ddf_list:
                ddf_list[key][:,t] = np.nan
    
    # compute errors
    for key in np.sort(ddf_list.keys()):
        print "Max error ddf[%s]: %.0f"%(key, np.nanmax(np.abs(ddf-ddf_list[key])))
        
    ncols = 2
    nsubplots = nf
    nrows = int(nsubplots/ncols)
    for key in np.sort(ddf_list.keys()):
        ff, ax = plt.subplots(nrows, ncols, sharex=True);
        ax = ax.reshape(nsubplots)
        for i in range(nf):
            ax[i].plot(time, ddf[i,:].A1,   '-',  label='Real')
            ax[i].plot(time, ddf_list[key][i,:].A1, '--', label='Approx '+key)
            ax[i].legend()
            ax[i].set_title('ddf '+str(i))            
        #    ax[i].set_ylabel('N/s*s')
        ax[-1].set_xlabel('Time [s]')
    
    if(PLOT_EIGENVALUES_J_Minv_ST_Mj_diag_JSTpinv):
        plt.figure()
        plt.plot(eig_J_Minv_ST_Mj_diag_JSTpinv.T)
        plt.title('Eigenvalues of J*Minv*S.T*Mj_diag*(J*S.T)^+')
    
    if(PLOT_DYNAMICS_RESIDUAL):
        ff, ax = plt.subplots(1, 1, sharex=True);
        for i in range(nv):
            ax.plot(time, dyn_res[i,:].A1,   '-',  label='Axis '+str(i))    
        ax.set_title('Dynamics residual')
        ax.set_ylabel('Nm')
        ax.set_xlabel('Time [s]')
        ax.legend()
    
    return ddf_list
    
    
def plot_stats_eigenvalues(eival, name=''):        
    print "STATISTICS OF EIGENVALUES OF SYSTEM %s"%(name)
    print "Max real part of eigenvalues of closed-loop linear system:      %.1f"%(np.nanmax(np.real(eival)))
    print "Max imaginary part of eigenvalues of closed-loop linear system: %.1f"%(np.nanmax(np.imag(eival)))
    print "Min real part of eigenvalues of closed-loop linear system:      %.1f"%(np.nanmin(np.real(eival)))
        
    plt.figure()
    plt.plot(np.real(eival).T)
    plt.title('Real part of eigenvalues of %s'%(name))
    
    plt.figure()
    plt.plot(np.imag(eival).T)
    plt.title('Imaginary part of eigenvalues of %s'%(name))
    
    return
    
def compute_closed_loop_transition_matrix(gains_array, robot, q, v, update=True):
    gains = Gains(gains_array)
    
    ny=3
    H = matlib.zeros((3*nf+2*ny, 3*nf+2*ny))
    H[:ny, ny:2*ny] = matlib.eye(ny)
    
    H_f = matlib.zeros((3*nf,3*nf))
    H_f[  :nf,     nf:2*nf] = matlib.eye(nf)
    H_f[nf:2*nf, 2*nf:3*nf] = matlib.eye(nf)
    H_f[2*nf:,   2*nf:3*nf] = -gains.kd_bar*matlib.eye(nf)
    
    if(update):
        se3.computeAllTerms(robot.model, robot.data, q, v)
        se3.framesForwardKinematics(robot.model, robot.data, q)
        
    M = robot.data.M        #(7,7)
    Jl,Jr = robot.get_Jl_Jr_world(q, False)
    J = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)    
    Mlf, Mrf = robot.get_Mlf_Mrf(q, False) 
    pyl, pzl = Mlf.translation[1:].A1
    pyr, pzr = Mrf.translation[1:].A1
    com, com_vel = robot.get_com_and_derivatives(q, v)
    cy, cz     = com.A1
    X_am  = np.matrix([-(pzl-cz),+(pyl-cy),-(pzr-cz),+(pyr-cy)])
    X_com = np.hstack([np.eye(2),np.eye(2)])
    X = np.vstack([X_com, X_am])
    X_pinv = np.linalg.pinv(X)
    
    Minv = np.linalg.inv(M)
    Upsilon = J*Minv*J.T
    S = matlib.zeros((na,nv))
    S[:,nv-na:] = matlib.eye(na)
    JSTpinv = np.linalg.pinv(J*S.T)        
    A = J*Minv*S.T*Mj_diag*JSTpinv
    
    # compute closed-loop transition matrix
    K1 = gains.kp_bar*K*A*gains.Kf
    K2 = K*Upsilon + gains.kp_bar*matlib.eye(nf)
    H_f[2*nf:,   1*nf:2*nf] = - K2
    H_f[2*nf:,   0*nf:1*nf] = - K1
    
    H[2*ny:,     2*ny:]        = H_f
    H[  ny:2*ny, 2*ny:2*ny+nf] = X
    H[ -nf:,         :ny]      = -K1*X_pinv*gains.Kp_com
    H[ -nf:,       ny:2*ny]    = -K1*X_pinv*gains.Kd_com
    return H
    
def compute_closed_loop_eigenvales(robot, gains_array):
    ''' Compute eigenvalues of linear part of closed-loop system:
            d3f = -Kd_bar*d2f - (K*Upsilon+Kp_bar)*df - Kp_bar*K*A*Kf*e_f + ...
    '''
    ny=3
    ei_cls_f = matlib.empty((3*nf,T), dtype=complex)*np.nan
    ei_cls   = matlib.empty((3*nf+2*ny,T), dtype=complex)*np.nan
    
    for t in range(T):
        H = compute_closed_loop_transition_matrix(gains_array, robot, q[:,t], v[:,t])
        H_f = H[2*ny:, 2*ny:]
        ei_cls_f[:,t] = np.sort_complex(eigvals(H_f)).reshape((3*nf,1))
        ei_cls[:,t] = np.sort_complex(eigvals(H)).reshape((3*nf+2*ny,1))
    
    plot_stats_eigenvalues(ei_cls_f, name='Closed-loop force tracking')
    plot_stats_eigenvalues(ei_cls, name='Closed-loop momentum tracking')
    return ei_cls
           
def cost_function(normalized_gains_array, robot, q0, v0, nominal_gains_array):
    MAX_REAL_EIG_VAL = -3
    MAX_NORMALIZED_GAIN = 2.0
    W_UNSTABLE = 1e3
    W_GAINS = 100.0

    gains_array = denormalize_gains_array(normalized_gains_array, nominal_gains_array);
    H = compute_closed_loop_transition_matrix(gains_array, robot, q0, v0, update=False);
    ei = eigvals(H);

    cost = 0.0
    for i in range(ei.shape[0]):
        ei_real, ei_imag = np.real(ei[i]), np.imag(ei[i])
        if(ei_real<0.0):
            cost += (ei_imag/ei_real)**2
            cost += max(ei_real, MAX_REAL_EIG_VAL)
        else:
            cost += W_UNSTABLE*(ei_real**2)
    
    for g in normalized_gains_array:
        if(g>MAX_NORMALIZED_GAIN):
            cost += W_GAINS*(g-MAX_NORMALIZED_GAIN)**2

    return cost
    
def step_response(H, dt=0.001, N=3000, plot=False):
    '''integral cost on the step responce'''
    n = H.shape[0];
    x0 = np.zeros(n)
    x0[0] = .1
    x = np.empty((N,n))
    x[0,:] = x0
    cost = 0.0
    e_dtH = expm(dt*H);
    for i in range(N-1):
        x[i+1,:] = np.dot(e_dtH, x[i,:]);
        cost += abs(x[i+1,0])
        
    if plot:
        max_rows = 4
        n_cols = 1 + (n+1)/max_rows
        n_rows = 1 + n/n_cols
        f, ax = plt.subplots(n_rows, n_cols, sharex=True);
        ax = ax.reshape(n_cols*n_rows)
        for i in range(n):
            ax[i].plot(x[:,i])
            ax[i].set_title(str(i))
        plt.show()
    return cost
    
def callback(x, f, accept):
    global nit
    print "%4d) Cost: %10.3f; Accept %d Gains:"%(nit, f, accept), x
    nit += 1;
    
def optimize_gains(robot, gains_array, q0, v0, niter=10):
    nominal_gains_array = copy.deepcopy(gains_array)
    H = compute_closed_loop_transition_matrix(gains_array, robot, q0, v0)
#    step_response(H, N=10000, plot=0)
    print "Initial gains:", gains_array.T
#    print "Initial normalized gains:", normalize_gains_array(gains_array, nominal_gains_array).T
    print "Initial eigenvalues:\n", np.sort_complex(eigvals(H)).T;
    print "Initial cost", cost_function(np.ones_like(gains_array), robot, q0, v0, nominal_gains_array)
    
    #optimize gain 
    global nit
    nit = 0
    opt_res = basinhopping(cost_function, np.ones_like(gains_array), niter, disp=False, T=0.1, stepsize=.01, 
                           minimizer_kwargs={'args':(robot, q0, v0, nominal_gains_array)}, callback=callback)
    opt_gains = denormalize_gains_array(opt_res.x, nominal_gains_array);
#    print opt_res, '\n'
    
    #Plot step response
    H = compute_closed_loop_transition_matrix(opt_gains, robot, q0, v0)
#    step_response(H, N=10000, plot=0)
    print "Optimal gains:      ", opt_gains
    print "Optimal normalized gains:", normalize_gains_array(opt_gains, nominal_gains_array).T
    print "Optimal eigenvalues:\n", list(np.sort_complex(eigvals(H)).T)
    return opt_gains

    
    
    
np.set_printoptions(precision=1, linewidth=200, suppress=True)

# User parameters
PLOT_DYNAMICS_RESIDUAL = 0
PLOT_EIGENVALUES_J_Minv_ST_Mj_diag_JSTpinv = 0

INPUT_FILE = os.getcwd()+'/../data/data_comparison_paper_v2/adm_ctrl_zeta_0.1_fDist_400.0/logger_data.npz'
dt  = 1e-3
ndt = 1
ZETA = 0.1
T_DISTURB_BEGIN = 0.10          # Time at which the disturbance starts
T_DISTURB_END   = 0.11          # Time at which the disturbance ends

# SETUP LOGGER
lgr = RaiLogger()
lgr.load(INPUT_FILE)
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
tau = lgr.get_vector('tsid_tau', na)
lkf_des = lgr.get_vector('tsid_lkf', nf/2)
rkf_des = lgr.get_vector('tsid_rkf', nf/2)
f_des = np.vstack((lkf_des,rkf_des))
q_cmd = lgr.get_vector('tsid_q_cmd', na)
lkf = lgr.get_vector('simu_lkf', nf/2)
rkf = lgr.get_vector('simu_rkf', nf/2)
f   = np.vstack((lkf,rkf))
df  = lgr.get_vector('simu_df', nf)
ddf = lgr.get_vector('simu_ddf', nf)

# shift ddf and dv 1 time step backward to align them with the state variables
dv[:,:-1]  = dv[:,1:]
dv[:,-1] = np.nan
ddf[:,:-1] = ddf[:,1:]
ddf[:,-1] = np.nan

#robot parameters
Ky, Kz = 23770., 239018.
By, Bz = ZETA*2*sqrt(Ky), ZETA*2*sqrt(Kz)
K     = np.asmatrix(np.diagflat([Ky,Kz,Ky,Kz]))
K_inv = np.linalg.inv(K)
#Controller parameters
Kf = np.matrix(np.diagflat([1.0/Ky, 1.0/Kz, 1.0/Ky, 1.0/Kz]))   # Stiffness of the feet spring
gains = Gains()
gains.Kp_adm, gains.Kd_adm, gains.Kp_com, gains.Kd_com, gains.Kf = 20.5603371308, 77.9184247381, 30.6694018561, 10.2970910213, 400*Kf # poles 5, 15, 25, 35
Kf_inv = np.linalg.inv(Kf)
se3.computeAllTerms(robot.model, robot.data, q[:,0], v[:,0])
Mj_diag = np.matrix(np.diag(np.diag(robot.data.M[3:,3:])))
gains.kp_bar = 1e4
gains.kd_bar = 200.0
gains.Kp_pos = gains.kp_bar*Mj_diag
gains.Kd_pos = gains.kd_bar*Mj_diag

#ddf_list = compute_ddf_approximations(robot)

#ei_cls = compute_closed_loop_eigenvales(robot, gains.to_array())

optimal_gains = optimize_gains(robot, gains.to_array(), q[:,0], v[:,0])
print "Optimal gains:\n", Gains(optimal_gains).to_string()

plt.show()
