# -*- coding: utf-8 -*-

""" 2018 Andrea Del Prete
"""

import math
import numpy as np
from numpy import dot, zeros, eye, ones
import scipy.linalg as linalg
from ExtendedKalmanFilter import ExtendedKalmanFilter

class MomentumEKF(ExtendedKalmanFilter):

    """ Implements an extended Kalman filter (EKF) for momentum estimation.
        The state vector is defined as:
            x = (c, dc, l, {f}, {df})
        where:
            c is the center of mass (CoM)
            dc is the CoM velocity
            l is the angular momentum
            {f} is the vector of contact forces
            {df} is the vector of contact force derivatives
        The estimator can work either in 2d or in 3d.
        The measurement vector is defined as:
            y = (c, dc, l, {f})
    
        Parameters
        ----------

        dt : float
            Time step
        mass : float
            Total mass of the system
        g : np.array
            Gravity acceleration vector
        c : np.array
            CoM position initial estimation
        dc : np.array
            CoM velocity initial estimation
        l : np.array
            Angular momentum initial estimation
        f : np.array
            Contact force initial estimation
        P : np.array
            Initial estimation covariance matrix
        sigma_c : np.array
            CoM position measurement noise std dev
        sigma_dc : np.array
            CoM velocity measurement noise std dev
        sigma_l : np.array
            Angular momentum measurement noise std dev
        sigma_f : np.array
            Contact force measurement noise std dev
        sigma_ddf : np.array
            Control noise std dev
    """
    
    USE_FINITE_DIFF = False # just for debug

    def __init__(self, dt, mass, g, c, dc, l, f, P, sigma_c,
                 sigma_dc, sigma_l, sigma_f, sigma_ddf):
        # compute size of state, measurements, and input
        self.n_lin = c.shape[0] # size of linear variables
        self.n_ang = l.shape[0] # size of angular variables
        self.n_f = f.shape[0] / self.n_lin # number of forces
        nl = self.n_lin
        na = self.n_ang
        nf = self.n_f
        dim_x = nl*2 + na + nl*nf*2
        dim_y = nl*2 + na + nl*nf
        dim_u = nl*nf
        super(MomentumEKF, self).__init__(dim_x, dim_y, dim_u)

        # compute indexes of state variables
        self.i_c  = 0
        self.i_dc = nl
        self.i_l  = 2*nl
        self.i_f  = 2*nl+na
        self.i_df = 2*nl+na+nf*nl
        
        # define shortcuts for indexes
        ic  = self.i_c
        idc = self.i_dc
        il  = self.i_l
        i_f = self.i_f
        idf = self.i_df
        
        # store date
        self.mass = mass
        self.g = g
        self.dt = dt
        dt2 = dt*dt
        dt3 = dt*dt2
        dt4 = dt*dt3
        self.P = P.copy()        # uncertainty covariance

        self.R[ic:ic+nl,   ic:ic+nl]   = np.diag(sigma_c**2)
        self.R[idc:idc+nl, idc:idc+nl] = np.diag(sigma_dc**2)
        self.R[il:il+na,   il:il+na]   = np.diag(sigma_l**2)
        self.R[i_f:,       i_f:]       = np.diag(sigma_f**2)
        
        # control transition matrix
        self.B = zeros((dim_x, dim_u))
        for i in range(nf):
            self.B[ic:idc,  i*nl:(i+1)*nl] = dt4*eye(nl)/(24*mass)
            self.B[idc:il,  i*nl:(i+1)*nl] = dt3*eye(nl)/(6*mass)
            self.B[il:i_f,  i*nl:(i+1)*nl] = dt3*np.ones((na,nl))    # this is not precise

        self.B[idf:,    :] = dt*eye(dim_u)
        self.B[i_f:idf, :] = 0.5*dt2*eye(dim_u)
        
        self.Q = dot(self.B, dot(np.diag(sigma_ddf**2), self.B.T)) # process uncertainty
          
        # initialize state estimation
        self.x = zeros(dim_x)
        self.x[ic:ic+nl]        = c
        self.x[idc:idc+nl]      = dc
        self.x[il:il+na]        = l
        self.x[i_f:i_f+nf*nl]   = f
        
        # fill measurement matrix
        self.H = zeros((dim_y,dim_x))
        self.H[:dim_y, :dim_y] = eye(dim_y);

        # compute process bias vector
        #self.b = zeros(dim_x)
        #self.b[ic: ic+nl]  = 0.5*dt2*g
        #self.b[idc:idc+nl] = dt*g
        
        # fill state transition matrix
        self.F = eye(dim_x)     # state transition matrix
        dt_I = dt*eye(nl)
        dt_sq_over_2m_I = 0.5*dt2/mass*eye(nl)
        dt_cu_over_6m_I = dt3*eye(nl)/(6*mass)
        self.F[ic: ic+nl, idc: idc+nl] = dt_I
        for i in range(nf):
            # fill rows of CoM position
            self.F[ic: ic+nl, i_f+i*nl: i_f+(i+1)*nl] = dt_sq_over_2m_I
            self.F[ic: ic+nl, idf+i*nl: idf+(i+1)*nl] = dt_cu_over_6m_I
            
            # fill rows of CoM velocity
            self.F[idc: idc+nl, i_f+i*nl: i_f+(i+1)*nl] = dt_I/mass
            self.F[idc: idc+nl, idf+i*nl: idf+(i+1)*nl] = dt_sq_over_2m_I

            # fill rows of contact forces
            self.F[i_f+i*nl: i_f+(i+1)*nl, idf+i*nl: idf+(i+1)*nl] = dt_I
    
    def get_state(self, as_matrices=False):
        '''
        Get the state vector, defined as:
            x = (c, dc, l, {f}, {df})
        '''
        # define shortcuts
        nl  = self.n_lin
        na  = self.n_ang
        nf  = self.n_f
        ic  = self.i_c
        idc = self.i_dc
        il  = self.i_l
        i_f = self.i_f
        idf = self.i_df
        
        # unpack state
        if(as_matrices):
            c  = np.asmatrix(self.x[ic:ic+nl]     ).T
            dc = np.asmatrix(self.x[idc:idc+nl]   ).T
            l  = np.asmatrix(self.x[il:il+na]     ).T
            f  = np.asmatrix(self.x[i_f:i_f+nf*nl]).T
            df = np.asmatrix(self.x[idf:idf+nf*nl]).T
        else:
            c  = self.x[ic:ic+nl]
            dc = self.x[idc:idc+nl]   
            l  = self.x[il:il+na]     
            f  = self.x[i_f:i_f+nf*nl]
            df = self.x[idf:idf+nf*nl]
        return (c, dc, l, f, df)
            
    def update_transition_matrix(self, p):
        ''' Update the transition matrix F
        
        Parameters
        ----------
        
        p : np.array
            Contact points
        '''
        assert(p.shape[0]==self.n_f*self.n_lin)
        
        if(self.USE_FINITE_DIFF):
            self.F = self.compute_dynamics_jac_by_fd(self.x, p)
            return
        
        # define shortcuts
        nl  = self.n_lin
        na  = self.n_ang
        nf  = self.n_f
        ic  = self.i_c
        idc = self.i_dc
        il  = self.i_l
        i_f = self.i_f
        idf = self.i_df
        dt  = self.dt
        
        # compute net force
        f  = self.x[i_f: i_f+nf*nl]
        df = self.x[idf: idf+nf*nl]
        f_net  = np.sum([ f[i*nl:(i+1)*nl] for i in range(nf)], axis=0)
        df_net = np.sum([df[i*nl:(i+1)*nl] for i in range(nf)], axis=0)
        
        # fill rows of angular momentum
        dt_fnet = dt*self.cross(f_net)
        half_dt_sq_df_net = 0.5*dt*dt*self.cross(df_net)
        self.F[il: il+na, ic: ic+nl]   = dt_fnet + half_dt_sq_df_net
        self.F[il: il+na, idc: idc+nl] = 0.5*dt*dt_fnet + 2*dt*half_dt_sq_df_net/3.0
        c  = self.x[ic:ic+nl]
        dc = self.x[idc:idc+nl]
        half_dt_sq_dc = 0.5*dt*dt*self.cross(dc)
        for i in range(nf):
            p_i = p[i*nl:(i+1)*nl]
            dt_pixc = dt*self.cross(p_i-c)
            # neglect terms proportional to dt^3
            self.F[il: il+na, i_f+i*nl: i_f+(i+1)*nl] = dt_pixc - half_dt_sq_dc
            self.F[il: il+na, idf+i*nl: idf+(i+1)*nl] = 0.5*dt*dt_pixc
            
    def predict_update(self, c, dc, l, f, p, ddf):
        """ Performs the predict/update innovation of the EKF.

        Parameters
        ----------

        c : np.array
            CoM position measurement for this step.
        dc : np.array
            CoM velocity measurement for this step.
        l : np.array
            Angular momentum measurement for this step.
        f : np.array
            Contact force measurements for this step.
        p : np.array
            Contact points for this step.
        ddf : np.array
            optional control vector input to the filter.
        """

        # compute measurement vector
        z = np.concatenate((c, dc, l, f))

        self.update_transition_matrix(p)
        
        F = self.F
        B = self.B
        H = self.H
        P = self.P
        Q = self.Q
        R = self.R

        # predict step
        self.x = self.predict_x(self.x, p, ddf)
        P = dot(F, P).dot(F.T) + Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(P)

        # update step
        PHT = dot(P, H.T)
        self.S = dot(H, PHT) + R
        self.K = dot(PHT, linalg.inv(self.S))

        self.y = z - dot(H, self.x)
        self.x = self.x + dot(self.K, self.y)

        I_KH = self._I - dot(self.K, H)
        self.P = dot(I_KH, P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)


    def compute_dynamics_jac_by_fd(self, x, p, u=0, delta=1e-5):
        x_0 = np.copy(x)
        f = self.predict_x(x, p, u)
        J = np.zeros((self.dim_x, self.dim_x))
        
        for i in range(self.dim_x):
            x_ = np.copy(x)
            x_[i] += delta
            f_ = self.predict_x(x_, p, u)
            J[:,i] = (f_ - f)/delta
        return J
            
    def predict_x(self, x, p, u=0):
        """
        Predicts the next state given the current state, the contact points, and the control. 
        """
        if np.isscalar(u):
            u = zeros(self.dim_u)
            
        # define shortcuts
        nl  = self.n_lin
        na  = self.n_ang
        nf  = self.n_f
        ic  = self.i_c
        idc = self.i_dc
        il  = self.i_l
        i_f = self.i_f
        idf = self.i_df
        m   = self.mass
        dt  = self.dt
        dt2 = dt*dt
        dt3 = dt*dt2
        dt4 = dt*dt3
        
        # unpack state
        c  = x[ic:ic+nl]     
        dc = x[idc:idc+nl]   
        l  = x[il:il+na]     
        f  = x[i_f:i_f+nf*nl]
        df = x[idf:idf+nf*nl]
        
        # compute net force
        f_net  = np.sum([f[i*nl:(i+1)*nl] for i in range(nf)], axis=0)
        df_net = np.sum([df[i*nl:(i+1)*nl] for i in range(nf)], axis=0)
        
        # compute accelerations, jerks and snaps
        ddc  = np.copy(self.g)
        d3c  = np.zeros(nl)
        d4c  = np.zeros(nl)
        dl   = np.zeros(na)
        ddl  = np.cross(f_net, dc)
        d3l  = 2*np.cross(df_net, dc)
        for i in range(nf):
            p_i   =  p[i*nl:(i+1)*nl]
            f_i   =  f[i*nl: (i+1)*nl]
            df_i  = df[i*nl: (i+1)*nl]
            ddf_i =  u[i*nl: (i+1)*nl]
            
            ddc  += f_i/m
            d3c  += df_i/m
            d4c  += ddf_i/m
            dl   += np.cross(p_i-c, f_i)
            ddl  += np.cross(p_i-c, df_i) # neglect dp_i
            d3l  += np.cross(p_i-c, ddf_i) #- 2*np.cross(dc, df_i) # neglect dp_i, ddp_i
        d3l += np.cross(f_net, ddc)
        
        # predict
        c_  = c  + dt*dc  + 0.5*dt2*ddc + dt3*d3c/6.0 + dt4*d4c/24.0
        dc_ = dc + dt*ddc + 0.5*dt2*d3c + dt3*d4c/6.0 
        l_  = l  + dt*dl  + 0.5*dt2*ddl + dt3*d3l/6.0
        f_  = f  + dt*df  + 0.5*dt2*u
        df_ = df + dt*u
        x_ = np.concatenate((c_, dc_, l_, f_, df_))        
        
        return x_
        
    def cross(self, p):
        if(p.shape[0]==2):
            return np.array([[-p[1], p[0]]])
        if(p.shape[0]==3):
            return np.array([[0.0, -p[2], p[1]],
                             [p[2], 0.0, -p[0]],
                             [-p[1], p[0], 0.0]]);
        assert(False)
        

