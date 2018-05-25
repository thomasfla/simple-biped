# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:34:19 2018

Test of EKF with linear 4-th order integrator.

@author: adelprete
"""
import sys
import os
sys.path += [os.getcwd()+'/..',]

from scipy import linalg as la
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from LQG_utils import lqr, dlqr, dkalman
from ExtendedKalmanFilter import ExtendedKalmanFilter
import matplotlib as mpl
mpl.rcParams['figure.figsize']      = 18, 9
np.set_printoptions(precision=2, linewidth=100, suppress=True)

def measurement(x):
    return x[:3]

def measurement_Jac(x):
    return C
    
def traj_sinusoid(t,start_position,stop_position,travel_time):
    # a cos(bt) + c
    A=-(stop_position-start_position)*0.5
    B = np.pi/travel_time
    C = start_position+(stop_position-start_position)*0.5
        
    p =         A*np.cos(B*t) + C
    v =      -B*A*np.sin(B*t)
    a =    -B*B*A*np.cos(B*t)
    j =   B*B*B*A*np.sin(B*t)
    s = B*B*B*B*A*np.cos(B*t) 
    return p,v,a,j,s
    
''' USER PARAMETERS '''
dt = 1e-2
T = 1000

w_x = 1.0           # state tracking cost
w_u = 1e-8          # control cost

sigma_w = 1e2      # process noise std dev
sigma_w_sim = 1e-0      # process noise std dev
sigma_v = np.array([1e-3, 1e-2, 1e-2]) # measurement noise std dev
sigma_x_0 = 1e1      # initial state estimate std dev

''' SYSTEM DYNAMICS '''
dt2 = dt**2
dt3 = dt*dt2
dt4 = dt*dt3
A = np.array([[1.0,  dt, 0.5*dt2, dt3/6.0],
              [0.0, 1.0,      dt, 0.5*dt2],
              [0.0, 0.0,     1.0,      dt],
              [0.0, 0.0,     0.0,     1.0]])
B = np.array([[dt4/24.0, dt3/6.0, dt2/2.0, dt]]).T
C = np.array([[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0]])
x_0 = np.zeros(4)

n_x = A.shape[0]
n_u = B.shape[1]
n_y = measurement(x_0).shape[0]

''' NOISE COVARIANCE MATRICES '''
W = sigma_w**2 * B.dot(B.T)
V = np.diag(sigma_v**2)
S_0 = sigma_x_0**2 * np.identity(n_x)

''' COST FUNCTION '''
Q = w_x * np.identity(n_x)
R = w_u * np.identity(n_u)

''' COMPUTE OPTIMAL LQR GAIN '''
(K_LQG,X,eigValsCtrl) = dlqr(A, B, Q, R)

''' CREATE EKF '''
ekf = ExtendedKalmanFilter(dim_x=n_x, dim_z=n_y, dim_u=n_u)
ekf.x = x_0 # initial state
ekf.P = S_0 # initial covariance
ekf.F = A   # transition matrix
ekf.B = B   # control matrix
ekf.R = V   # measurement noise
ekf.Q = W   # process noise

''' COMPUTE GAIN OF STEADY-STATE KALMAN FILTER '''
(L_ss, S_ss, eigValsObs)  = dkalman(A, C, W, V)

''' SIMULATE SYSTEM '''
w     = normal(np.zeros((T,n_x)), sigma_w_sim, (T, n_x))
v     = normal(np.zeros((T,n_y)), sigma_v, (T, n_y))
x     = np.zeros((T, n_x))
x_hat = np.zeros((T, n_x))
x_hat_ss = np.zeros((T, n_x))
y     = np.zeros((T, n_y))
u     = np.zeros((T-1, n_u))
x[0,:]     = x_0
x_hat[0,:] = x_0 + w[0,:]
x_hat_ss[0,:] = x_hat[0,:]
y[0,:]     = measurement(x[0,:]) + v[0,:]
for t in range(T-1):
    #(p,dp,a,j,u[t,0])  = traj_sinusoid(t*dt, x_0[0], 1.0, 1.0)
    #u[t,:]      -= np.dot(K_LQG, x_hat[t,:] - np.array([p,dp,a,j]))
    x[t+1,:]     = A.dot(x[t,:])   + B.dot(u[t,:]) + w[t,:]
    y[t+1,:]     = measurement(x[t+1,:]) + v[t+1,:]
    ekf.predict_update(y[t+1,:], measurement_Jac, measurement, u=u[t,:])
    x_hat[t+1,:] = ekf.x
    
    ''' steady-state KF '''
    x_predict       = A.dot(x_hat_ss[t,:]) + B.dot(u[t,:])
    x_hat_ss[t+1,:] = x_predict + L_ss.dot(y[t+1,:] - C.dot(x_predict))

''' PLOT RESULTS '''
f, ax = plt.subplots(4,1,sharex=True);
for i in range(4):
    ax[i].plot(x[:,i],         '-',   label='x')
    if(i<0):
        ax[i].plot(y[:,i],         '--',  label='y')
    ax[i].plot(x_hat[:,i],     '-',   label='x EKF')
    ax[i].plot(x_hat_ss[:,i],  '-',   label='x KF ss')
    ax[i].grid();
    ax[i].legend();
    ax[i].set_title('State '+str(i))
    

plt.show()
