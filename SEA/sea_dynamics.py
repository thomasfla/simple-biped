import sys
import numpy as np
import numpy.linalg as la
from scipy.linalg import expm
from numpy.linalg import eigvals
from math import sqrt
    
class SEA:
    ''' A series elastic actuator (SEA) with the following dynamics
            tau_j = k*(q_m-q_j)
            tau_j = I_j*ddq_j + b_j*dq_j
            tau_m = I_m*ddq_m + b_m*dq_m + tau_j

        Defining the system state as x=(q_j, dq_j, q_m, dq_m),
        the linear system dynamics is then:
            dq_j  = dq_j
            ddq_j = I_j^-1 * (k*(q_m-q_j) - b_j*dq_j)
            dq_m  = dq_m
            ddq_m = I_m^-1 * (- k*(q_m-q_j) - b_m*dq_m) + I_m^-1 * tau_m
    
        Defining the system state as x=(q_j, dq_j, tau_j, dtau_j),
        the linear system dynamics is then:
            dq_j    = dq_j
            ddq_j   = I_j^-1 * (tau_j - b_j*dq_j)
            dtau_j  = dtau_j
            ddtau_j = k*(ddq_m - ddq_j) = 
                    =  k*I_m^-1*(-tau_j-b_m*dq_m) 
                      -k*I_j^-1*( tau_j-b_j*dq_j) + k*I_m^-1*tau_m = 
                    =  k*I_m^-1*(-tau_j-b_m*(k^-1*dtau_j+dq_j)) 
                      -k*I_j^-1*( tau_j-b_j*dq_j) + k*I_m^-1*tau_m = 
                    =  k*I_m^-1*(-tau_j - b_m*k^-1*dtau_j - b_m*dq_j) 
                      +k*I_j^-1*(-tau_j + b_j*dq_j) + k*I_m^-1*tau_m = 
                    = -k*(I_m^-1+I_j^-1)*tau_j 
                      -I_m^-1*b_m*dtau_j 
                      -k*(I_m^-1*b_m+I_j^-1*b_j)*dq_j 
                      +k*I_m^-1*tau_m
    '''
    USE_TAU_AS_STATE_VARIABLE = 0
    
    def __init__(self, dt, k, I_j, b_j, I_m, b_m):
        self.dt  = dt;
        self.k   = k;
        self.I_j = I_j;
        self.b_j = b_j;
        self.I_m = I_m;
        self.b_m = b_m;

        self.x = np.zeros(4);

        # compute system matrices in continuous time
        Ijinv = 1.0/I_j;
        Iminv = 1.0/I_m;
        if(self.USE_TAU_AS_STATE_VARIABLE):
            a31 = -k*(Iminv*b_m + Ijinv*b_j)
            a32 = -k*(Iminv+Ijinv)
            self.A = np.array([[     0.0,        1.0,      0.0,        0.0],
                               [     0.0, -Ijinv*b_j,    Ijinv,        0.0],
                               [     0.0,        0.0,      0.0,        1.0],
                               [     0.0,        a31,      a32, -Iminv*b_m]]);
            self.B = np.array([0.0, 0.0, 0.0, k*Iminv]).T;
        else:
            self.A = np.array([[     0.0,        1.0,      0.0,        0.0],
                               [-Ijinv*k, -Ijinv*b_j,  Ijinv*k,        0.0],
                               [     0.0,        0.0,      0.0,        1.0],
                               [ Iminv*k,        0.0, -Iminv*k, -Iminv*b_m]]);
            self.B = np.array([0.0, 0.0, 0.0, Iminv]).T;
        
        # convert to discrete time
        H = np.zeros((5,5))
        H[:4,:4] = dt*self.A
        H[:4,4]  = dt*self.B
        expH = expm(H)
        
        self.Ad = expH[:4,:4]
        self.Bd = expH[:4,4]
        
        #self.Ad = expm(dt*self.A);
        #A_cond = la.cond(self.A)
        #if A_cond < 1/sys.float_info.epsilon:
            # if A is invertible then:
            #   int_0^dt e^{t*A} dt = A^-1 * (e^(dt*A) - I)
        #    self.Bd = la.inv(self.A).dot(self.Ad - np.eye(4)).dot(self.B)
        #else:
        #    print "Condition number of matrix A is too large to try to invert A:", A_cond
        
    def simulate(self, tau_m):
        self.x = self.Ad.dot(self.x) + self.Bd.dot(tau_m)
        return self.x
        
        dx = self.A.dot(self.x) + self.B.dot(tau_m)
        print 'dx=', dx.T #(self.A.dot(self.x)).T, ' +B*u', (self.B.dot(tau_m)).T
        
        q_j  = self.x[0]
        dq_j = self.x[1]
        q_m  = self.x[2]
        dq_m = self.x[3]
        k    = self.k
        I_j  = self.I_j
        b_j  = self.b_j
        I_m  = self.I_m
        b_m  = self.b_m
        
        tau_j = k*(q_m-q_j)
        ddq_j = (tau_j - b_j*dq_j)/I_j
        ddq_m = (- tau_j - b_m*dq_m + tau_m)/I_m
        
        print 'ddq_j=%.2f \tddq_m=%.2f \ttau_j=%.2f'%(ddq_j, ddq_m, tau_j)
        
        self.x += self.dt*dx
        
     
    def qj(self):
        return self.x[0];
        
    def dqj(self):
        return self.x[1];
           
    def tau(self):
        if(self.USE_TAU_AS_STATE_VARIABLE):
            return self.x[2];
        return self.k*(self.x[2]-self.x[0]);
        
    def dtau(self):
        if(self.USE_TAU_AS_STATE_VARIABLE):
            return self.x[3];
        return self.k*(self.x[3]-self.x[1])
