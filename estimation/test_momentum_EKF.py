# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:34:19 2018

Test of momentum EKF

@author: adelprete
"""

from scipy import linalg as la
import numpy as np
from numpy import ones, zeros, dot
import matplotlib.pyplot as plt
from numpy.random import normal
from momentumEKF import MomentumEKF
import matplotlib as mpl

np.set_printoptions(precision=2, linewidth=100, suppress=True)
mpl.rcParams['figure.figsize']      = 18, 8
    
def RMSE(s1,s2):
    return np.sqrt(np.mean((s1-s2)**2))
    
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
TEST_DYNAMICS_LINEARIZATION_BY_FINITE_DIFF = 0
dt = 1e-2
T = 500                              # number of time steps
mass = 100                          # robot mass
g = np.array([0.0, -9.81])          # gravity acc
p = np.array([0.3, 0.0, -0.3, 0.0]) # contact points

sigma_x_0 = 1e0              # initial state estimate std dev
sigma_ddf = 1e2*ones(4)      # control (i.e. force accelerations) noise std dev used in EKF
sigma_ddf_sim = 1e2*ones(4)  # control (i.e. force accelerations) noise std dev used in simulation
sigma_c  = 1e-3*ones(2)      # CoM position measurement noise std dev
sigma_dc = 1e-2*ones(2)      # CoM velocity measurement noise std dev
sigma_l  = 1e0*ones(1)      # angular momentum measurement noise std dev
sigma_f  = 1e-2*mass*ones(4)       # force measurement noise std dev

''' INITIAL STATE '''
c_0 = np.array([0.0, 0.5])
dc_0 = np.zeros(2)
l_0 = np.zeros(1)
f_0 = np.array([0.0, -0.5*mass*g[1], 0.0, -0.5*mass*g[1]])
df_0 = zeros(4)

n_x = 9+4
n_u = 4
n_y = 9

''' CREATE EKF '''
S_0 = sigma_x_0**2 * np.eye(n_x)
ekf = MomentumEKF(dt, mass, g, c_0, dc_0, l_0, f_0, S_0, sigma_c, sigma_dc, sigma_l, sigma_f, sigma_ddf)

''' COPY MATRICES FROM EKF '''
V = ekf.R.copy() # measurement noise covariance
B = ekf.B.copy() # control matrix
C = ekf.H.copy() # measurement matrix

''' SIMULATE SYSTEM '''
w     = normal(0.0, sigma_ddf_sim, (T, n_u))
v     = normal(np.zeros((T,n_y)), np.sqrt(np.diag(V)), (T, n_y))
x     = np.zeros((T, n_x))
x_prior = np.zeros((T, n_x))
x_hat = np.zeros((T, n_x))
y     = np.zeros((T, n_y))
u     = np.zeros((T-1, n_u))
x_0   = np.concatenate((c_0, dc_0, l_0, f_0, df_0))
x[0,:]       = x_0
x_hat[0,:]   = x_0 + dot(B, w[0,:])
x_prior[0,:] = x_hat[0,:]
y[0,:]       = dot(C, x[0,:]) + v[0,:]
for t in range(T-1):
    (pp,dp,a,j,u[t,0]) = traj_sinusoid(t*dt, x[0,0], 0.3, 1.0)
    (pp,dp,a,j,u[t,1]) = traj_sinusoid(t*dt, x[0,1], 0.8, 1.0)
    u[t,:] *= mass
    u[t,2:4] = -u[t,0:2]
    #u[t,:]       = -np.dot(K_LQG, x_hat[t,:])
    
    x[t+1,:]     = ekf.predict_x(x[t,:], p, u[t,:]) + dot(B, w[t,:])
    y[t+1,:]     = dot(C, x[t+1,:]) + v[t+1,:]
    c  = y[t+1, :2]
    dc = y[t+1, 2:4]
    l  = y[t+1, 4:5]
    f  = y[t+1, 5:]
    ekf.predict_update(c, dc, l, f, p, u[t,:])
    if(t==0):
        print "Max EKF gain at time 0: ", np.max(np.abs(ekf.K))
    x_hat[t+1,:] = ekf.x
    x_prior[t+1,:] = ekf.x_prior

print "Max EKF gain at time T: ", np.max(np.abs(ekf.K)), "\n" #, ekf.K

''' TEST DYNAMICS LINEARIZATION BY FINITE DIFFERENCES '''
if(TEST_DYNAMICS_LINEARIZATION_BY_FINITE_DIFF):
    F_fd  = ekf.compute_dynamics_jac_by_fd(ekf.x, p, zeros(n_u))
    ekf.update_transition_matrix(p)
    F = np.copy(ekf.F)
    E = F - F_fd
    print "TEST DYNAMICS LINEARIZATION BY FINITE DIFFERENCES:"
    print "F-F_fd=", np.max(np.abs(ekf.F-F_fd)), "\n"

''' COMPUTE ESTIMATION RMSE '''
c_hat  = x_hat[:,:2]
dc_hat = x_hat[:,2:4]
l_hat  = x_hat[:,4]
f_hat  = x_hat[:,5:9]
df_hat = x_hat[:,9:]
c  = x[:,:2]
dc = x[:,2:4]
l  = x[:,4]
f  = x[:,5:9]
df = x[:,9:]
print "RMSE c:  ", RMSE(c, c_hat)
print "RMSE dc: ", RMSE(dc, dc_hat)
print "RMSE l:  ", RMSE(l, l_hat)
print "RMSE f:  ", RMSE(f, f_hat)
print "RMSE df: ", RMSE(df, df_hat)


''' PLOT RESULTS '''
f, ax = plt.subplots(2,1,sharex=True);
for i in range(2):
    ax[i].plot(x[:,i],         '-',  label='x')
#    ax[i].plot(y[:,i],         '--',  label='y')
    ax[i].plot(x_hat[:,i],     '-',  label='x EKF')
    ax[i].plot(x_prior[:,i],     '--',  label='x EKF prior')
    ax[i].grid();
    ax[i].legend();
    ax[i].set_title('CoM Pos '+str(i))

f, ax = plt.subplots(2,1,sharex=True);
for i in range(2):
    ax[i].plot(x[:,2+i],         '-',  label='x')
#    ax[i].plot(y[:,2+i],         '--',  label='y')
    ax[i].plot(x_hat[:,2+i],     '-',  label='x EKF')
    ax[i].plot(x_prior[:,2+i],   '--',  label='x EKF prior')
    ax[i].grid();
    ax[i].legend();
    ax[i].set_title('CoM Vel '+str(i))
    
f, ax = plt.subplots(1,1,sharex=True);
ax.plot(x[:,4],         '-',  label='x')
#ax.plot(y[:,4],         '--',  label='y')
ax.plot(x_hat[:,4],     '-',  label='x EKF')
ax.grid();
ax.legend();
ax.set_title('Ang mom')

f, ax = plt.subplots(2,2,sharex=True);
ax = ax.reshape(4)
for i in range(4):
    ax[i].plot(x[:,5+i],         '-',  label='x')
    #ax[i].plot(y[:,5+i],         '--',  label='y')
    ax[i].plot(x_hat[:,5+i],     '-',  label='x EKF')
    ax[i].plot(x_prior[:,5+i],   '--',  label='x EKF prior')
    ax[i].grid();
    ax[i].legend();
    ax[i].set_title('Forces '+str(i))
    
f, ax = plt.subplots(2,2,sharex=True);
ax = ax.reshape(4)
for i in range(4):
    ax[i].plot(x[:,9+i],         '-',  label='x')
    ax[i].plot(x_hat[:,9+i],     '-',  label='x EKF')
    ax[i].plot(x_prior[:,9+i],   '--',  label='x EKF prior')
    ax[i].grid();
    ax[i].legend();
    ax[i].set_title('Force derivatives '+str(i))
    
f, ax = plt.subplots(2,2,sharex=True);
ax = ax.reshape(4)
for i in range(4):
    ax[i].plot(u[:,i],         '-',  label='u')
    ax[i].grid();
    ax[i].legend();
    ax[i].set_title('Force 2nd derivatives '+str(i))

plt.show()
