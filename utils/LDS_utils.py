# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:35:03 2018

Utility functions related to Linear Dynamical Systems (LDS).

@author: adelprete
"""

from numpy import matlib
import numpy as np
from numpy.linalg import norm, eigvals, eig
from scipy.linalg import expm
import matplotlib.pyplot as plt

from pinocchio.utils import *
import simple_biped.utils.plot_utils as plut

from math import pi,sqrt
import os
np.set_printoptions(precision=1, linewidth=200, suppress=True)

class Empty:
    pass
    
def simulate_ALDS(H, x0, dt, N, plot=False, show_plot=None):
    '''Simulate an Autonomous Linear Dynamical System (ALDS) forward in time 
        H: transition matrix
        x0: initial state
        dt: time step
        N: number of time steps to simulate
        plot: if True it plots the time evolution of the state
    '''
    n = H.shape[0];
    x = matlib.empty((n,N))
    x[:,0] = x0
    e_dtH = expm(dt*H)
    for i in range(N-1):
        x[:,i+1] = e_dtH * x[:,i]
        
    if plot:
        max_rows = 4
        n_cols = 1 + (n+1)/max_rows
        n_rows = 1 + n/n_cols
        f, ax = plt.subplots(n_rows, n_cols, sharex=True);
        ax = ax.reshape(n_cols*n_rows)
        time = np.arange(N*dt, step=dt)
        for i in range(n):
            ax[i].plot(time, x[i,:].A1)
            ax[i].set_title(str(i))
        if(show_plot is None):
            plt.show()
    return x

def compute_quadratic_state_integral_ALDS(H, x0, T, dt=None):
    ''' Assuming the state x(t) evolves in time according to a linear dynamic:
            dx(t)/dt = H * x(t)
        Compute the following integral:
            int_{0}^{T} x(t)^T * x(t) dt
    '''
    if(dt is None):
        w, V = eig(H) # H = V*matlib.diagflat(w)*V^{-1}
        print "Eigenvalues H:", np.sort_complex(w).T
        Lambda_inv = matlib.diagflat(1.0/w)
        e_2T_Lambda = matlib.diagflat(np.exp(2*T*w))
        int_e_2T_Lambda = 0.5*Lambda_inv*(e_2T_Lambda - matlib.eye(n))
    #    V_inv = np.linalg.inv(V)
    #    cost = x0.T*(V_inv.T*(int_e_2T_Lambda*(V_inv*x0)))
        V_inv_x0 = np.linalg.solve(V, x0)
        cost = V_inv_x0.T*int_e_2T_Lambda*V_inv_x0
        return cost[0,0]
        
    N = int(T/dt)
    x = simulate_ALDS(H, x0, dt, N)
    cost = 0.0
    not_finite_warning_printed = False
    for i in range(N):
        if(np.all(np.isfinite(x[:,i]))):
            cost += dt*(x[:,i].T * x[:,i])[0,0]
        elif(not not_finite_warning_printed):
            print 'WARNING: x is not finite at time step %d'%(i) #, x[:,i].T
            not_finite_warning_printed = True
    return cost
    
def compute_weighted_quadratic_state_integral_ALDS(H, x0, T, Q, Q_inv=None, dt=None):
    ''' Assuming the state x(t) evolves in time according to a linear dynamic:
            dx(t)/dt = H * x(t)
        Compute the following integral:
            int_{0}^{T} x(t)^T * Q^2 * x(t) dt
        where the matrix Q must be full rank.
    '''
    if(dt is None):
        if(Q_inv is None):
            Q_inv = np.linalg.inv(Q)
        H_bar = Q * H * Q_inv
        y0 = Q*x0
        return compute_quadratic_state_integral_ALDS(H_bar, y0, T)

    if(Q_inv is None):
        Q_inv = np.linalg.inv(Q)
    H_bar = Q * H * Q_inv
    y0 = Q*x0
    return compute_quadratic_state_integral_ALDS(H_bar, y0, T, dt)
        
    N = int(T/dt)
    x = simulate_ALDS(H, x0, dt, N)
    cost = 0.0
    not_finite_warning_printed = False
    Q2 = Q*Q
    for i in range(N):
        if(np.all(np.isfinite(x[:,i]))):
            cost += dt*(x[:,i].T * Q2 * x[:,i])[0,0]
        elif(not not_finite_warning_printed):
            print 'WARNING: x is not finite at time step %d'%(i) #, x[:,i].T
            not_finite_warning_printed = True
    return cost
    
def compute_matrix_exponential_integral(H, T, dt=None):
    ''' Compute the following integral:
            int_{0}^{T} e^{t*H} dt
    '''
    n = H.shape[0]
    if(dt is None):
        w, V = eig(H) 
        # H      = V*diagflat(w)*V^{-1}
        # H^{-1} = V*diagflat(1.0/w)*V^{-1}
        V_inv = np.linalg.inv(V)
        H_inv = V*matlib.diagflat(1.0/w)*V_inv
        e_TH = V*matlib.diagflat(np.exp(T*w))*V_inv
        return H_inv * (e_TH - matlib.eye(n))
        
    N = int(T/dt)    
    res = matlib.zeros((n,n))
    for i in range(N):
        res += dt*expm(i*dt*H)
    return res
    
def compute_integral_e_t_Lambda_Q_e_t_Lambda(Lambda_diag, Q_diag, T, dt=None):
    ''' Compute the following integral:
            int_{0}^{T} e^{t*Lambda}*Q*e^{t*Lambda} dt
        where both Lambda and Q are diagonal matrices
    '''
    n = Lambda_diag.shape[0]
    if(dt is None):
        # H      = V*diagflat(w)*V^{-1}
        # H^{-1} = V*diagflat(1.0/w)*V^{-1}
        Lambda_inv = matlib.diagflat(1.0/Lambda_diag)
        e_2_T_Lambda = matlib.diagflat(np.exp(2*T*Lambda_diag))
        Q = matlib.diagflat(Q_diag)
        return 0.5 * Q * Lambda_inv * (e_2_T_Lambda - matlib.eye(n))
        
    N = int(T/dt)
    res = matlib.zeros((n,n))
    Lambda = matlib.diagflat(Lambda_diag)
    Q = matlib.diagflat(Q_diag)
    for i in range(N):
        e_t_Lambda = expm(i*dt*Lambda)
        res += dt*(e_t_Lambda*Q*e_t_Lambda)
    return res
    
if __name__=='__main__':
    T = 2.0
    dt = 1e-3
    n = 4
    x0 = matlib.rand((n,1))
    H  = matlib.rand((n,n))
    H  = - H*H.T
    Q_diag = matlib.rand(n)
#    Q_diag = matlib.ones(n)
    
    ei = eigvals(H);
    print "Eigenvalues of transfer matrix H:", np.sort_complex(ei).T    
    print "Q_diag:", Q_diag
    print "x0:", x0.T
    
#    int_e_tH_approx = compute_matrix_exponential_integral(H, T, dt=1e-3)    
#    int_e_tH        = compute_matrix_exponential_integral(H, T)    
#    print "Integral of e^{tH}:\n", int_e_tH
#    print "Approximated integral of e^{tH}:\n", int_e_tH_approx
#    print "Approximation error:", np.max(np.abs(int_e_tH_approx-int_e_tH)), '\n'
#    
#    Lambda_diag = matlib.diag(H)
#    int_e_t_Lambda_Q_e_t_Lambda        = compute_integral_e_t_Lambda_Q_e_t_Lambda(Lambda_diag, Q_diag, T)
#    int_e_t_Lambda_Q_e_t_Lambda_approx = compute_integral_e_t_Lambda_Q_e_t_Lambda(Lambda_diag, Q_diag, T, 1e-3)
#    print "integral e^{t*Lambda}*Q*e^{t*Lambda}:\n", int_e_t_Lambda_Q_e_t_Lambda
#    print "Approximated integral e^{t*Lambda}*Q*e^{t*Lambda}:\n", int_e_t_Lambda_Q_e_t_Lambda_approx
#    print "Approximation error:", np.max(np.abs(int_e_t_Lambda_Q_e_t_Lambda_approx-int_e_t_Lambda_Q_e_t_Lambda)), '\n'
    
    cost_approx = compute_quadratic_state_integral_ALDS(H, x0, T, dt)
    cost        = compute_quadratic_state_integral_ALDS(H, x0, T)
    print "Approximated cost is:", cost_approx
    print "Exact cost is:       ", cost
    print "Approximation error: ", abs(cost-cost_approx), '\n'

    Q = matlib.diagflat(Q_diag)
    Q_inv = matlib.diagflat(1.0/Q_diag)
    cost_approx = compute_weighted_quadratic_state_integral_ALDS(H, x0, T, Q, Q_inv, dt)
    cost        = compute_weighted_quadratic_state_integral_ALDS(H, x0, T, Q, Q_inv)
    print "Approximated cost is:", cost_approx
    print "Exact cost is:       ", cost
    print "Approximation error: ", abs(cost-cost_approx)
    
    