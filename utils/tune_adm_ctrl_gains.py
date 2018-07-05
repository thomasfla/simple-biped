# -*- coding: utf-8 -*-
"""
Script to find the gains for an admittance controller.

Created on Thu Jul  5 14:41:32 2018

@author: adelprete
"""

import numpy as np
from numpy.linalg import eigvals
from scipy.signal import place_poles

def compute_system_dynamics(K):
    H = np.array( [[0,         1,        0,       0],
                   [0,         0,        1,       0],
                   [0,         0,        0,       1],
                   [-K[0],  -K[1],   -K[2],   -K[3]]])
    A = np.array( [[0,         1,        0,       0],
                   [0,         0,        1,       0],
                   [0,         0,        0,       1],
                   [0.,       0.,       0.,      0.]])
    B = np.array([[0., 0., 0., 1.]]).T
    return (H, A, B);

def compute_system_dynamics_dt(K, dt):
    dt2 = dt*dt/2.0
    dt3 = dt*dt2/3.0
    dt4 = dt*dt3/4.0
    A = np.array( [[1,        dt,      dt2,     dt3],
                   [0,         1,       dt,     dt2],
                   [0,         0,        1,      dt],
                   [0,         0,        0,       1]])
    B = np.array([[dt4, dt3, dt2, dt]]).T
    H = A - np.outer(B,K)
    return (H, A, B);
    
# INPUT PARAMETERS
dt = 1e-3
#des_gains = np.array([1.20e+06, 1.54e+05, 7.10e+03, 1.40e+02])    # desired gains of 4-th order closed-loop system
#des_gains = np.array([52674.83686644, 13908.30537877,  1377.10995895, 60.5999991])
p1 = -15.0
dp = -0.1
des_poles = np.array([p1, p1+dp, p1+2*dp, p1+3*dp])
#K = 23770.      # contact stiffness
K = 239018.
DISCRETE_POLE_PLACEMENT = 1

if('des_gains' not in locals()):
    if(DISCRETE_POLE_PLACEMENT):
        print "Computing gains corresponding to desired poles in discrete time"
        (H, A, B) = compute_system_dynamics_dt(np.zeros(4), dt)
        des_poles = np.exp(des_poles*dt)
    else:
        print "Computing gains corresponding to desired poles in continuous time"
        (H, A, B) = compute_system_dynamics(np.zeros(4))
    res = place_poles(A, B, des_poles)
    des_gains = res.gain_matrix.squeeze()
    print "Desired gains:", des_gains

# COMPUTE ADMITTANCE CONTROL GAINS
(K1, K2, K3, K4) = des_gains.tolist()
Kf = 1.0/K              # force feedback gain
Kd_adm = K4                 # contact point velocity gain
Kp_adm = K3 / (1+K*Kf)      # contact point position gain
Kd_com = K2 / (K*Kf*Kp_adm)    # CoM derivative gain
Kp_com = K1 / (K*Kf*Kp_adm)    # CoM position gain

(H, A, B) = compute_system_dynamics(des_gains);
ei = eigvals(H);
(H_dt, A, B) = compute_system_dynamics_dt(des_gains, dt);
ei_dt = eigvals(H_dt);
print "\nEigenvalues corresponding to desired closed-loop gains of 4-th order system:\n", ei
print "\nEigenvalues of discrete-time closed-loop system:\n", ei_dt
print "\nCorresponding gains for admittance control:"
for gain in ['Kf', 'Kp_adm', 'Kd_adm', 'Kp_com', 'Kd_com']:
    print gain+'=', locals()[gain]