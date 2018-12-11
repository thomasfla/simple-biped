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
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

from pinocchio.utils import *
from simple_biped.admittance_ctrl import GainsAdmCtrl
from simple_biped.hrp2_reduced import Hrp2Reduced
from simple_biped.simu import Simu
from simple_biped.utils.LDS_utils import simulate_ALDS
from simple_biped.robot_model_path import pkg, urdf 
import simple_biped.utils.plot_utils as plut

np.set_printoptions(precision=1, linewidth=200, suppress=True)

class Empty:
    pass


class GainOptimizer(object):
    
    def __init__(self):
        self.nit = 0
        self.initial_cost = 1.0
    
    def callback(self, x, f, accept):
        print "%4d) Cost: %5f; Accept %d Gains:"%(self.nit, f/self.initial_cost, accept), x
        self.nit += 1;

    def optimize_gains(self, gains, cost_function, niter=100, verbose=0):
        ''' Optimize the gains of a controller based on the cost function
        '''
        self.initial_cost = cost_function(gains)
        if(verbose):            
            print "Initial gains:   ", gains.T
            print "Initial cost     ", self.initial_cost
        self.nit = 0
        opt_res = basinhopping(cost_function, gains, niter, disp=False, T=0.1, stepsize=.01, callback=self.callback)
        opt_gains = opt_res.x
        if(verbose):
            print "Optimal gains:   ", opt_gains
            print "Optimal cost     ", cost_function(opt_gains)
        return opt_gains


class GainOptimizeAdmCtrl(GainOptimizer):
    
    def __init__(self, robot, q, v, K, ny, nf, nominal_gains, dt, x0, N, Q):
        self.robot = robot
        self.K = K      # contact stiffness
        self.ny = ny    # size of momentum vector
        self.nf = nf    # size of contact force
        self.nominal_gains = nominal_gains  # initial guess for gains
        self.dt = dt    # controller time step
        self.x0 = x0    # initial state for cost function evaluation
        self.N = N      # number of time steps for cost function evaluation
        self.Q = Q      # cost function matrix: x^T * Q * x
        
        self.nv = robot.model.nv
        self.na = self.nv-ny
        
        self.H = matlib.zeros((3*nf+2*ny, 3*nf+2*ny))
        self.H[:ny, ny:2*ny] = matlib.eye(ny)
        
        self.H_f = matlib.zeros((3*nf,3*nf))
        self.H_f[  :nf,     nf:2*nf] = matlib.eye(nf)
        self.H_f[nf:2*nf, 2*nf:3*nf] = matlib.eye(nf)
        
        self.update_state(q, v)
        
        self.nominal_cost = self.cost_function(np.ones_like(nominal_gains))
        
    def set_cost_function_matrix(self, Q):
        self.Q = Q
    
    def normalize_gains_array(self, gains_array):
        return np.divide(gains_array, self.nominal_gains)
    
    def denormalize_gains_array(self, normalized_gains_array):
        return np.multiply(normalized_gains_array, self.nominal_gains)
    
    def update_state(self, q, v):
        nf, ny = self.nf, self.ny
        se3.computeAllTerms(self.robot.model, self.robot.data, q, v)
        se3.framesForwardKinematics(self.robot.model, self.robot.data, q) # replace with updateFramePlacements(model, data)
        M = self.robot.data.M        #(7,7)
        Mj_diag = np.matrix(np.diag(np.diag(M[ny:,ny:])))
        Jl,Jr = self.robot.get_Jl_Jr_world(q, False)
        J = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)    
        Mlf, Mrf = self.robot.get_Mlf_Mrf(q, False) 
        pyl, pzl = Mlf.translation[1:].A1
        pyr, pzr = Mrf.translation[1:].A1
        com, com_vel = self.robot.get_com_and_derivatives(q, v)
        cy, cz     = com.A1
        X_am  = np.matrix([-(pzl-cz),+(pyl-cy),-(pzr-cz),+(pyr-cy)])
#        X_com = np.hstack([np.eye(2)/M[0,0],np.eye(2)/M[0,0]])
        X_com = np.hstack([np.eye(2),np.eye(2)])
        X = np.vstack([X_com, X_am])
        self.X_pinv = np.linalg.pinv(X)
        
        self.H[  ny:2*ny, 2*ny:2*ny+nf] = X
        
        Minv = np.linalg.inv(M)
        Upsilon = J*Minv*J.T
        S = matlib.zeros((self.na, self.nv))
        S[:,self.nv-self.na:] = matlib.eye(self.na)
        JSTpinv = np.linalg.pinv(J*S.T)        
        A = J*Minv*S.T*Mj_diag*JSTpinv
        self.K_A = self.K*A
        self.K_Upsilon = self.K*Upsilon
        
    def compute_transition_matrix(self, gains_array):
        gains = GainsAdmCtrl(gains_array)
        nf, ny = self.nf, self.ny
        self.H_f[2*nf:,   2*nf:3*nf] = -gains.kd_bar*matlib.eye(nf)
        
        # compute closed-loop transition matrix
        K1 = gains.kp_bar*self.K_A*gains.Kf
        K2 = self.K_Upsilon + gains.kp_bar*matlib.eye(nf)
        self.H_f[2*nf:,   1*nf:2*nf] = - K2
        self.H_f[2*nf:,   0*nf:1*nf] = - K1
        
        self.H[2*ny:,     2*ny:]        = self.H_f
        self.H[ -nf:,         :ny]      = -K1*self.X_pinv*gains.Kp_com
        self.H[ -nf:,       ny:2*ny]    = -K1*self.X_pinv*gains.Kd_com
        return self.H
        
    def cost_function(self, normalized_gains):
        gains_array = self.denormalize_gains_array(normalized_gains);
        H = self.compute_transition_matrix(gains_array);
        x = simulate_ALDS(H, self.x0, self.dt, self.N, 0)
        cost = 0.0
        not_finite_warning_printed = 1
        for i in range(self.N):
            if(np.all(np.isfinite(x[:,i]))):
                last_finite_x = x[:,i]
            elif(not not_finite_warning_printed):
                print 'WARNING: x is not finite at time step %d'%(i) #, x[:,i].T
                not_finite_warning_printed = True
            cost += self.dt*(last_finite_x.T * self.Q * last_finite_x)[0,0]
        cost /= (self.N*self.dt)
#        print 'Gains:', GainsAdmCtrl(gains_array).to_string()
#        print "Largest eigenvalues:", np.sort_complex(eigvals(H))[-4:].T,
#        print 'Cost ', cost
        if not np.isfinite(cost):
            print 'WARNING: cost is not finite', cost
#            step_response(H, self.x0, self.dt, self.N, 1)
        return cost
        
    def normalized_cost_function(self, normalized_gains):
        cost = self.cost_function(normalized_gains)
        return cost/self.nominal_cost
        
    def optimize_gains(self, niter=100):
        initial_guess = np.ones_like(self.nominal_gains)
        
        opt_gains_normalized = super(GainOptimizeAdmCtrl, self).optimize_gains(initial_guess, self.normalized_cost_function, niter)
        opt_gains = self.denormalize_gains_array(opt_gains_normalized)
        
        return opt_gains

def convert_cost_function(w_x, w_dx, w_d2x, w_d3x, w_d4x):
    ''' Convert the cost function of the CoM state in the cost function
        of the momentum+force state.
    '''
    nl = 2
    ny = 3
#    robot   = Hrp2Reduced(urdf, [pkg], loadModel=0, useViewer=0)
    m       = 1.0 #np.sum([robot.model.inertias[i].mass for i in range(1,len(robot.model.inertias))])
    Q_diag  = np.matrix(nl*[w_x] + nl*[w_dx] + nl*[w_d2x] + nl*[w_d3x] + nl*[w_d4x])
    Q_c     = matlib.diagflat(Q_diag)                           # weight matrix for CoM state
    S       = np.hstack((np.eye(nl), np.zeros((nl,ny-nl))))     # matrix selecting CoM from x
    SA      = np.hstack((np.eye(nl), np.eye(nl)))               # matrix adding forces
    P       = np.matrix((1.0/m)*block_diag(S, S, SA, SA, SA))   # projection from full state to CoM state
    Q       = P.T * Q_c * P
    return Q
    
def optimize_gains_adm_ctrl(Q, N, dt, max_iter, x0=None, initial_guess=None, do_plots=0):
    # User parameters
    PROFILE_ON = 0
    ny = 3
    nf = 4
    ss = 3*nf+2*ny
    if x0 is None:
        x0 = matlib.zeros((ss,1))
        x0[0,0] = 1.0
    
    # SETUP
    robot   = Hrp2Reduced(urdf, [pkg], loadModel=0, useViewer=0)
    q       = robot.q0.copy()
    v       = zero(robot.model.nv)
    K       = Simu.get_default_contact_stiffness()
    
    if(initial_guess is None):
        initial_guess   = GainsAdmCtrl.get_default_gains(K).to_array()
    
    gain_optimizer = GainOptimizeAdmCtrl(robot, q, v, K, ny, nf, initial_guess, dt, x0, N, Q)
    
    if(do_plots):
        H = gain_optimizer.compute_transition_matrix(initial_guess);
        simulate_ALDS(H, x0, dt, N, 1)
    # OPTIMIZE GAINS
    
    if PROFILE_ON:
        import cProfile
        cProfile.run('optimal_gains = gain_optimizer.optimize_gains(2)');
    else:
        optimal_gains = gain_optimizer.optimize_gains(max_iter)
        
    if(do_plots):
        H = gain_optimizer.compute_transition_matrix(optimal_gains);
        simulate_ALDS(H, x0, dt, N, 1)
        
    return optimal_gains


if __name__=='__main__':
    N       = 1000
    dt      = 1e-2
    w_x         = 1.0
    w_dx        = 1e-1 #0.0
    w_d2x       = 1e-3
    w_d3x       = 1e-6
    w_d4x       = 1e-9
    max_iter = 2
    do_plots = 1
    ny = 3
    nf = 4
    x0          = matlib.zeros((3*nf+2*ny,1))
    x0[ny,0]    = .0765     # initial CoM velocity in Y direction
    
    Q = convert_cost_function(w_x, w_dx, w_d2x, w_d3x, w_d4x)
    optimal_gains = optimize_gains_adm_ctrl(Q, N, dt, max_iter, x0=x0, do_plots=do_plots)
    
    print "Initial gains:\n", GainsAdmCtrl.get_default_gains(Simu.get_default_contact_stiffness()).to_string()
    print "Optimal gains:\n", GainsAdmCtrl(optimal_gains).to_string()