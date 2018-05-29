# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:34:19 2018

Simple 1d test of EKF using linear system dynamics, but nonlinear measurements.

@author: adelprete
"""

from scipy import linalg as la
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from LQG_utils import lqr, dlqr, dkalman
from ExtendedKalmanFilter import ExtendedKalmanFilter
import matplotlib as mpl
mpl.rcParams['figure.figsize']      = 18, 8

def measurement(x):
    #return np.sin(x)
    return np.array(x.dot(x)*(x));

def measurement_Jac(x):
    #return np.cos(x)
    #return 2*np.array([[x[0]]]);
    return 3*np.array([[x[0]**2]]);
    
''' USER PARAMETERS '''
dt = 1e-2
T = 5000
N = 1               # number of simulations to run

w_x = 1.0           # state tracking cost
w_u = 1e-2          # control cost

sigma_w = 1e-2      # process noise std dev
sigma_v = 1e-3      # measurement noise std dev
sigma_x_0 = 1e-1      # initial state estimate std dev

''' SYSTEM DYNAMICS '''
A = np.array([[1.0]])
B = np.array([[dt]])
x_0 = np.array([1.7])

n_x = A.shape[0]
n_u = B.shape[1]
n_y = measurement(x_0).shape[0]

''' NOISE COVARIANCE MATRICES '''
W = sigma_w**2 * np.identity(n_x)
V = sigma_v**2 * np.identity(n_y)
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


''' SIMULATE SYSTEM '''
w     = normal(np.zeros((T,n_x)), sigma_w, (T, n_x))
v     = normal(np.zeros((T,n_y)), sigma_v, (T, n_y))
x     = np.zeros((T, n_x))
x_hat = np.zeros((T, n_x))
y     = np.zeros((T, n_y))
u     = np.zeros((T-1, n_u))
x[0,:]     = x_0
x_hat[0,:] = x_0 + w[0,:]
y[0,:]     = measurement(x[0,:]) + v[0,:]
for t in range(T-1):
    u[t,:]       = -np.dot(K_LQG, x_hat[t,:])
    x[t+1,:]     = A.dot(x[t,:])   + B.dot(u[t,:]) + w[t,:]
    y[t+1,:]     = measurement(x[t+1,:]) + v[t+1,:]
    ekf.predict_update(y[t+1,:], measurement_Jac, measurement, u=u[t,:])
    x_hat[t+1,:] = ekf.x

''' PLOT RESULTS '''
f, ax = plt.subplots(1,1,sharex=True);
ax.plot(x,         '-',  label='x')
ax.plot(y,         '-',  label='y')
ax.plot(x_hat,     '--',  label='x EKF')
ax.grid();
ax.legend();
plt.title('State')

plt.show()
