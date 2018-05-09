# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:34:19 2018

@author: adelprete
"""

from scipy import linalg as la
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
#import pinocchio_inv_dyn.plot_utils as plot_utils

import matplotlib as mpl
mpl.rcParams['figure.figsize']      = 23, 12
 
def lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.
     
    dx/dt = A x + B u
     
    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151 
    #first, try to solve the ricatti equation
    X = la.solve_continuous_are(A, B, Q, R)
    #compute the LQR gain
    K = la.solve(R, B.T.dot(X))
    eigVals, eigVecs = la.eig(A-B.dot(K))
    return K, X, eigVals
 
def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.     
     
    x[k+1] = A x[k] + B u[k]
     
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151 
    #first, try to solve the ricatti equation
    X = la.solve_discrete_are(A, B, Q, R)
    #compute the LQR gain
    K = la.solve(B.T.dot(X).dot(B)+R, B.T.dot(X).dot(A))    
    eigVals, eigVecs = la.eig(A-B.dot(K))     
    return K, X, eigVals
    
def dkalman(A,C,W,V):
    """Solve the infinite-horizon discrete-time Kalman observer.     
     
    x[k+1] = A x[k] + B u[k] + w[y]
    y[k]   = C x[k] + v[t]
     
    """
    #first, try to solve the ricatti equation
    S = la.solve_discrete_are(A.T, C.T, W, V)
    #compute the Kalman gain
    L = la.solve(C.dot(S).dot(C.T)+V, C.dot(S).dot(A.T)).T
    eigVals, eigVecs = la.eig(A-L.dot(C))
    return L, S, eigVals

def simulate(A, B, C, x_0, K, L, w=None, v=None):
    n_x = A.shape[0]
    n_u = B.shape[1]
    n_y = C.shape[0]
    x     = np.zeros((T, n_x))
    x_hat = np.zeros((T, n_x))
    y     = np.zeros((T, n_y))
    u     = np.zeros((T-1, n_u))
    if(w is None):
        w     = normal(np.zeros((T,n_x)), sigma_w, (T, n_x))
    if(v is None):
        v     = normal(np.zeros((T,n_y)), sigma_v, (T, n_y))
    x[0,:]     = x_0
    x_hat[0,:] = x_0 + w[0,:]
    for t in range(T-1):
        u[t,:]       = -np.dot(K, x_hat[t,:])
        x[t+1,:]     = A.dot(x[t,:])     + B.dot(u[t,:]) + w[t,:]
        y[t+1,:]     = C.dot(x[t+1,:]) + v[t+1,:]
        x_pred       = A.dot(x_hat[t,:]) + B.dot(u[t,:])
        x_hat[t+1,:] = x_pred + L.dot(y[t+1,:] - C.dot(x_pred))
    return (x, x_hat, y, u)
    
def compute_cost(x, u, Q, R):
    c_x = 0.0
    c_u = 0.0
    for t in range(x.shape[0]-1):
        c_x += x[t,:].T.dot(Q).dot(x[t,:])
        c_u += u[t,:].T.dot(R).dot(u[t,:])
    t = x.shape[0]-1
    c_x += x[t,:].T.dot(Q).dot(x[t,:])
    return (c_x+c_u, c_x, c_u)
    
''' USER PARAMETERS '''
dt = 1e-2
T = 10000
N = 1               # number of simulations to run

w_x = 1.0           # state tracking cost
w_u = 1e-2          # control cost

sigma_w = 1e-1      # process noise std dev
sigma_v = 1e-1      # measurement noise std dev
sigma_x_0 = 1e-2      # initial state estimate std dev

L_PD = np.array([[0.618]])          # low pass filter coefficient
K_PD = np.array([[5.0]])         # proportional gain

''' SYSTEM DYNAMICS '''
A = np.array([[1.0]])
B = np.array([[dt]])
C = np.array([[1.0]])
x_0 = np.array([.1])

n_x = A.shape[0]
n_u = B.shape[1]
n_y = C.shape[0]

''' NOISE COVARIANCE MATRICES '''
W = sigma_w**2 * np.identity(n_x)
V = sigma_v**2 * np.identity(n_y)
S_0 = sigma_x_0**2 * np.identity(n_x)

''' COST FUNCTION '''
Q = w_x * np.identity(n_x)
R = w_u * np.identity(n_u)

''' COMPUTE OPTIMAL LQR AND KALMAN GAINS '''
(K_LQG,X,eigValsCtrl) = dlqr(A, B, Q, R)
(L_LQG,S,eigValsObs)  = dkalman(A, C, W, V)
print "LQG controller gain:\n", K_LQG
print "LQG observer gain:\n", L_LQG
print "PD controller gain:\n", K_PD
print "PD observer gain:\n", L_PD

''' SIMULATE SYSTEM '''
for i in range(N):
    w_large     = normal(np.zeros((T,n_x)), sigma_w, (T, n_x))
    v_large     = normal(np.zeros((T,n_y)), sigma_v, (T, n_y))
#    w_large     = normal(np.zeros((T,n_x)), 1e2*sigma_w, (T, n_x))
#    v_large     = normal(np.zeros((T,n_y)), 1e2*sigma_v, (T, n_y))
    (x_LQG, x_hat_LQG, y_LQG, u_LQG) = simulate(A, B, C, x_0, K_LQG, L_LQG, w_large, v_large)
    (x_PD,  x_hat_PD,  y_PD,  u_PD)  = simulate(A, B, C, x_0, K_PD,  L_PD,  w_large, v_large)
    (cost_LQG, cost_LQG_x, cost_LQG_u) = compute_cost(x_LQG, u_LQG, Q, R)
    (cost_PD,  cost_PD_x,  cost_PD_u)  = compute_cost(x_PD,  u_PD,  Q, R)
    print "Cost LQG %.3f (x: %.3f, u %.3f)" % (cost_LQG, cost_LQG_x, cost_LQG_u)
    print "Cost PD  %.3f (x: %.3f, u %.3f)" % (cost_PD,  cost_PD_x,  cost_PD_u)

''' PLOT RESULTS '''
plt.plot(u_LQG, label='u LQG')
plt.plot(u_PD, label='u PD')
plt.legend()
plt.title('Control')
plt.grid()

#f, ax = plt.subplots(2,1,sharex=True);
#ax[0].plot(x_LQG,     '-',  label='x LQG')
#ax[0].plot(x_hat_LQG, '--', label='x hat LQG')
#ax[0].plot(y_LQG,     ':',  label='y LQG')
#ax[0].grid()
#ax[1].plot(x_PD,      '-',  label='x PD')
#ax[1].plot(x_hat_PD,  '--', label='x hat PD')
#ax[1].plot(y_PD,      ':',  label='y PD')
#ax[1].grid()
#ax[0].legend();
#ax[1].legend();
#plt.title('State')

f, ax = plt.subplots(1,1,sharex=True);
ax.plot(x_LQG,     '-',  label='x LQG')
ax.plot(x_PD,      '-',  label='x PD')
ax.grid();
ax.legend();
plt.title('State')

plt.show()