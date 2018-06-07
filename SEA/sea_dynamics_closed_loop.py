import sys
import numpy as np
import numpy.linalg as la
from scipy.linalg import expm
from numpy.linalg import eigvals
from math import sqrt
from sea_dynamics import SEA, ImpedanceControl
        
class CloseLoopSEA:
    ''' A series elastic actuator (SEA) with the following dynamics
            tau_j = k*(q_m-q_j)
            tau_j = I_j*ddq_j + b_j*dq_j
            tau_m = I_m*ddq_m + b_m*dq_m + tau_j
        in closed loop with the following controller:
            tau_d = -kp*q_j - Kd*dq_j
            tau_m = tau_d - k_tau*(tau_j-tau_d) - b_tau*dtau_j
        The state is defined as x = (qj, dqj, ddqj, dddqj)
    '''
    
    def __init__(self, dt, k, I_j, b_j, I_m, b_m, k_p, k_d, k_tau, b_tau, use_ff=True):
        self.t   = 0.0
        self.use_ff = use_ff
        self.dt  = dt;
        self.k   = k;
        self.I_j = I_j;
        self.b_j = b_j;
        self.I_m = I_m;
        self.b_m = b_m;
        self.k_p = k_p
        self.k_d = k_d
        self.k_tau = k_tau
        self.b_tau = b_tau

        self.x = np.zeros(4);

        # compute coefficients of characteristic polynomial:
        # p(s) = c4*s^4 + c3*s^3 + c2*s^2 + c1*s + c0
        c4 = I_m * I_j / k
        c3 = (I_m*b_j + I_j*b_m)/k + I_j*b_tau
        c2 = b_m*b_j/k + I_j + I_m + I_j*k_tau + b_j*b_tau
        c1 = b_j + b_m + b_j*k_tau + k_tau*k_d
        c0 = k_tau*k_p
        
        if(use_ff):
            c1 += k_d
            c0 += k_p
        
        # compute feedback gains of 4-th order LDS
        self.k0 = c0/c4 # k_tau, k_p
        self.k1 = c1/c4 # k_tau, k_d
        self.k2 = c2/c4 # k_tau, b_tau
        self.k3 = c3/c4 # b_tau
        
        # compute coefficients such that characteristic polynomial can be written as
        # p(s) = s^4 + (a0*b_tau + a1)*s^3 + 
        #        (a2*k_tau + a3*b_tau + a4)*s^2 + 
        #        (a5*k_tau + a6*(k_tau+1)*kd + a7)*s + a8*(k_tau+1)*kp
        self.a0 = I_j / c4
        self.a1 = ((I_m*b_j + I_j*b_m)/k) / c4
        self.a2 = I_j / c4
        self.a3 = b_j / c4
        self.a4 = (b_m*b_j/k + I_j + I_m) / c4
        self.a5 = b_j / c4
        self.a6 = 1.0 / c4
        self.a7 = (b_j + b_m) / c4
        self.a8 = 1.0 / c4
        
        # compute continuous state space transfer matrix
        self.A = np.array([[     0.0,        1.0,      0.0,        0.0],
                           [     0.0,        0.0,      1.0,        0.0],
                           [     0.0,        0.0,      0.0,        1.0],
                           [  -c0/c4,     -c1/c4,    -c2/c4,    -c3/c4]]);
        
        # convert to discrete time
        self.Ad = expm(dt*self.A)
                
    def simulate(self):
        self.t += self.dt
        self.x = self.Ad.dot(self.x)
        return self.x
        
    def max_eigen_value(self):
        ei = eigvals(self.A);
        return np.max(np.real(ei))
        
    def is_stable(self):
        return self.max_eigen_value()<0.0

    def check_RH_conditions(self):
        (a0, a1, a2, a3, a4, a5, a6, a7, a8, k0, k1, k2, k3) = self.get_shortcuts()
        if(k0<=0.0 or k1<=0.0 or k2<=0.0 or k3<=0.0):
            return False;
        if(k1 >= k2*k3):
            return False;
        if(k0 >= k1*k2/k3 - k1*k1/(k3*k3)):
            return False
        return True
        
    def compute_lambda_bounds(self):
        ''' Assuming kp=lambda^2 and kd=2*lambda, compute the bounds
            on lambda given the current values of k_tau and b_tau.
            This computation is based on the RH stability criterion.
        '''
        (a0, a1, a2, a3, a4, a5, a6, a7, a8, k0, k1, k2, k3) = self.get_shortcuts()
        k_tau = self.k_tau

        b1 = a5*k_tau + a7
        if(self.use_ff):
            a_bar = (k_tau+1)*(a8 + 4*a6*a6*(k_tau+1)/(k3**2))
            b_bar = (2*a6*(k_tau+1)/k3) * (2*b1/k3 - k2)
        else:
            a_bar = k_tau*(a8 + 4*a6*a6*k_tau/(k3**2))
            b_bar = (2*a6*k_tau/k3) * (2*b1/k3 - k2)
        
        c_bar = (b1/k3)**2 - k2*b1/k3    
        delta = b_bar**2 - 4*a_bar*c_bar
        if(delta<0.0):
            # system unstable for any value of lambda
            print "SEA unstable for any value of lambda"
            return [0.0, -1.0]
        # bounds based on 2nd RH condition k0 < k1*k2/k3 - (k1/k3)^2
        lb = (-b_bar - sqrt(delta)) / (2*a_bar)
        ub1 = (-b_bar + sqrt(delta)) / (2*a_bar)
        
        # upper bound based on first RH condition k1<k2*k3
        ub2 = (-a5*k_tau - a7 + k2*k3) / (2*a6*(k_tau+1))
        return [lb, min(ub1, ub2)];
        
    def get_shortcuts(self):
        ''' Utility method to call to define all useful local shortcuts:
            (a0, a1, a2, a3, a4, a5, a6, a7, a8, k0, k1, k2, k3) = self.get_shortcuts()
        '''
        return (self.a0, self.a1, self.a2, self.a3, self.a4, self.a5, self.a6,
                self.a7, self.a8, self.k0, self.k1, self.k2, self.k3)
        
    def qj(self):
        return self.x[0];
        
    def dqj(self):
        return self.x[1];
        
    def tau(self):
        # tau_j = I_j*ddq_j + b_j*dq_j
        return self.I_j*self.x[2] + self.b_j*self.x[1];
        
    def dtau(self):
        # dtau_j = I_j*dddq_j + b_j*ddq_j
        return self.I_j*self.x[3] + self.b_j*self.x[2];
        

class CloseLoopDtSEA:
    ''' A series elastic actuator (SEA) with the following dynamics
            tau_j = k*(q_m-q_j)
            tau_j = I_j*ddq_j + b_j*dq_j
            tau_m = I_m*ddq_m + b_m*dq_m + tau_j
        in closed loop with the following discrete-time controller:
            tau_d = -kp*q_j - Kd*dq_j
            tau_m = -k_tau*(tau_j-tau_d) - b_tau*dtau_j
        The state is defined as x = (qj, dqj, qm, dqm)
    '''
    
    def __init__(self, dt, k, I_j, b_j, I_m, b_m, k_p, k_d, k_tau, b_tau, use_ff=True):
        self.t   = 0.0
        self.dt  = dt;
        self.use_ff = use_ff
        self.sea  = SEA(dt, k, I_j, b_j, I_m, b_m)
        self.ctrl = ImpedanceControl(k_p, k_d, k_tau, b_tau, k, use_ff)
        self.x = self.sea.x
        self.k_tau = k_tau
        self.b_tau = b_tau
        self.k = k
        
        # compute discrete-time state space transfer matrix
        self.H = self.sea.Ad + np.outer(self.sea.Bd, self.ctrl.K)

                
    def simulate(self):
        self.t += self.dt
        self.x = self.H.dot(self.x)
        return self.x
        
    def max_eigen_value(self):
        ei = eigvals(self.H);
        return np.max(np.abs(ei))
        
    def is_stable(self):
        return self.max_eigen_value()<1.0
        
    def _is_stable_for_kp_kd(self, k_p, k_d):
        ctrl = ImpedanceControl(k_p, k_d, self.k_tau, self.b_tau, self.k, self.use_ff)
        H = self.sea.Ad + np.outer(self.sea.Bd, ctrl.K)
        ei = eigvals(H);
        if(np.max(np.abs(ei))>1.0):
            return False
        return True
        
    def compute_lambda_bounds(self, tol=1e-3, max_iter=1000):
        ''' Assuming kp=lambda^2 and kd=2*lambda, compute the bounds
            on lambda given the current values of k_tau and b_tau.
            This computation is based on the eigen values of the closed-loop
            matrix and a binary search.
        '''
        # initial bounds on lambda_max
        ub_min = 0.0
        ub_max = 1e10
        
        lmbda = ub_max
        stable_once = False
        for i in range(max_iter):
            k_p = lmbda**2
            k_d = 2.0*lmbda
            if(not self._is_stable_for_kp_kd(k_p, k_d)):
                ub_max = lmbda # system unstable => update upper bound
            else:
                stable_once = True
                ub_min = lmbda # system stable => update lower bound
            
            # check termination condition
            if(ub_max-ub_min<tol):
                if(not stable_once):
                    #print 'System never stable with k_tau=%f, b_tau=%f'%(self.k_tau, self.b_tau)
                    return (np.nan, np.nan)
                    
                #print "Lambda max found in %d iter: %f"%(i, ub_min)
                return (0.0, 0.5*(ub_max+ub_min))
                
            if(i==max_iter-1):
                print "Could not find solution in %d iter"%(i)
                
            lmbda = 0.5*(ub_max+ub_min)

        return (0.0, 0.5*(ub_max+ub_min))
        
    def qj(self):
        return self.x[0];
        
    def dqj(self):
        return self.x[1];
        
    def tau(self):
        # tau_j = I_j*ddq_j + b_j*dq_j
        return self.sea.I_j*self.x[2] + self.sea.b_j*self.x[1];
        
    def dtau(self):
        # dtau_j = I_j*dddq_j + b_j*ddq_j
        return self.sea.I_j*self.x[3] + self.sea.b_j*self.x[2];
