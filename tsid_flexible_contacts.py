import pinocchio as se3
from pinocchio.utils import *
from math import pi,sqrt,cos,sin
from quadprog import solve_qp
import matplotlib.pyplot as plt
from numpy import matlib
from utils.tsid_utils import createContactForceInequalities
from pinocchio_inv_dyn.acc_bounds_util_multi_dof import computeAccLimits, compute_min_ddq_stop, areStatesViable, computeVelViabBounds, compute_min_q_upper_bound
import numpy as np

try:
    from IPython import embed
except ImportError:
    pass

class Empty:
    pass
    
class GainsTsidFlexK:
    ''' Class collecting the gains of TSID-Flex-K:
        (Kp_mom, Kd_mom, Ka_mom, Kj_mom)
    '''
    
    def __init__(self, gain_array=None):
        '''
            gain_array: numpy array containing all the gains
        '''
        if gain_array is None:
            gain_array = np.zeros(4)
        self.from_array(gain_array)
    
    def to_array(self):
        res = np.zeros(4)
        res[0] = self.Kp
        res[1] = self.Kd
        res[2] = self.Ka
        res[3] = self.Kj
        return res
        
    def from_array(self, gains):
        self.Kp = gains[0]
        self.Kd = gains[1]
        self.Ka = gains[2]
        self.Kj = gains[3]
        
    def to_string(self):
        res = ''
        for s in ['Kp', 'Kd', 'Ka', 'Kj']:
            res += s+' = '+str(self.__dict__[s])+'\n'
        return res
        
    @staticmethod
    def get_default_gains(K):
        gains = GainsTsidFlexK()
        #    (Kp_com, Kd_com, Ka_com, Kj_com) = (10611.05989124,  4182.20596787,   618.10999684,    40.5999999)     # poles [-10.3 -10.2 -10.1  -10.]
        #    (Kp_com, Kd_com, Ka_com, Kj_com) = (52674.83686644, 13908.30537877,  1377.10995895, 60.5999991) # poles [-15.30235117 -15.19247204 -15.10739361 -14.99778229]
        #    (Kp_com, Kd_com, Ka_com, Kj_com) = (1.64844157e+05, 3.27244115e+04, 2.43611027e+03, 8.06000045e+01) # ploes [-20.29999909 -20.20000804 -20.09999556 -20.00000185]
        #    (Kp_com, Kd_com, Ka_com, Kj_com) = (2.28323600e+05, 4.76834812e+04, 3.35391925e+03, 9.68316039e+01) # poles 10, 20, 30, 40
        #    (Kp_com, Kd_com, Ka_com, Kj_com) = (7.78064620e+05, 1.03624061e+05, 5.18844741e+03, 1.16188573e+02) # poles -30
        #    (Kp_com, Kd_com, Ka_com, Kj_com) = (1.20e+06, 1.54e+05, 7.10e+03, 1.40e+02) # poles [-50. -40. -30. -20.]
        gains.Kp, gains.Kd, gains.Ka, gains.Kj = 63057.32417634, 21171.16628651,  2076.59405021,    77.91842474 # poles 5, 15, 25, 35
        return gains
        
        
class TsidFlexibleContact:
    HESSIAN_REGULARIZATION = 1e-8
    
    def __init__(self,robot, Ky, Kz, w_post, Kp_post, gains, fMin, mu, ddf_max, dt, estimator = None):
        self.robot = robot
        self.NQ = robot.model.nq
        self.NV = robot.model.nv
        self.NB = robot.model.nbodies
        self.RF = robot.model.getFrameId('rankle')
        self.LF = robot.model.getFrameId('lankle')
        self.RK = robot.model.frames[self.RF].parent
        self.LK = robot.model.frames[self.LF].parent
        
        self.estimator = estimator
        self.w_post = w_post
        self.Kp_post = Kp_post
        self.Kd_post = 2*sqrt(Kp_post)
        self.Kp_com = gains.Kp
        self.Kd_com = gains.Kd
        self.Ka_com = gains.Ka
        self.Kj_com = gains.Kj
        self.Ky = Ky
        self.Kz = Kz
        self.fMin = fMin
        
        self.data = Empty()
        self.data.ddf_des = matlib.zeros(4)
        self.data.com_s_des = np.matrix([0.,0.]).T.A1 #need for update the com estimator at first iteration
        com_p_ref = np.matrix([0.,0.53]).T
        com_v_ref = np.matrix([0.,0.]).T
        com_a_ref = np.matrix([0.,0.]).T
        com_j_ref = np.matrix([0.,0.]).T
        com_s_ref = np.matrix([0.,0.]).T
        self.callback_com = lambda t: (com_p_ref,com_v_ref,com_a_ref,com_j_ref,com_s_ref )
        
        self.post_p_ref = robot.q0
        self.post_v_ref = matlib.zeros(7).T
        self.post_a_ref = matlib.zeros(7).T
        self.A_post     = w_post*matlib.eye(7, 7+4+4)
        self.Kspring    = -np.matrix(np.diagflat([Ky,Kz,Ky,Kz]))   # Stiffness of the feet spring

        self.mu = mu
        self.ddf_max = ddf_max
        self.dt = dt        
        (B, b) = createContactForceInequalities(self.mu, self.fMin) # B_f * f <= b_f
        k = B.shape[0]
        self.B_f = matlib.zeros((2*k,4));
        self.b_f = matlib.zeros(2*k).T;
        self.B_f[:k,:2] = B
        self.B_f[k:,2:] = B
        self.b_f[:k,0] = b
        self.b_f[k:,0] = b
        
        self.ind = []
     
    def _compute_com_task(self, t, com_est, com_v_est, com_a_est, com_j_est):
        Kp_com,  Kd_com, Ka_com, Kj_com  = self.Kp_com, self.Kd_com, self.Ka_com, self.Kj_com
        com_p_ref, com_v_ref, com_a_ref, com_j_ref, com_s_ref = self.callback_com(t)
        com_p_err = com_est   - com_p_ref
        com_v_err = com_v_est - com_v_ref
        com_a_err = com_a_est - com_a_ref
        com_j_err = com_j_est - com_j_ref
        com_s_des = -Kp_com*com_p_err -Kd_com*com_v_err -Ka_com*com_a_err -Kj_com*com_j_err +com_s_ref
        
        m = self.robot.data.mass[0]
        Xc = (1./m) * np.hstack([np.eye(2),np.eye(2)])
        Nc = np.matrix([0.,0.]).T
        
        self.data.com_p_err = com_p_err.A1
        self.data.com_v_err = com_v_err.A1
        self.data.com_a_err = com_a_err.A1
        self.data.com_j_err = com_j_err.A1
        self.data.comref    = com_p_ref.A1
        
        return com_s_des, Xc, Nc
        
    def _compute_ang_mom_task(self, q, v, am_est, com_est, com_v_est, com_a_est, f_est, df_est):
        Mlf, Mrf = self.robot.get_Mlf_Mrf(q, False)
        pyl, pzl = Mlf.translation[1:].A1
        pyr, pzr = Mrf.translation[1:].A1
        fyl, fzl, fyr, fzr = f_est.A1
        cy, cz     = com_est.A1
        dcy, dcz   = com_v_est.A1
        ddcy, ddcz = com_a_est.A1  
        dfyl, dfzl, dfyr, dfzr = df_est.A1

        Ky, Kz = self.Ky, self.Kz
        dpyl, dpzl, dpyr, dpzr = -dfyl/Ky, -dfzl/Kz, -dfyr/Ky, -dfzr/Kz
        Xl = np.matrix([ 1./-Ky*fzl - (pzl - cz),
                        -1./-Kz*fyl + (pyl - cy),
                         1./-Ky*fzr - (pzr - cz),
                        -1./-Kz*fyr + (pyr - cy)])
        Nl  = fyl*ddcz - fzl*ddcy + 2*(dpyl-dcy)*dfzl - 2*(dpzl-dcz)*dfyl #checked
        Nl += fyr*ddcz - fzr*ddcy + 2*(dpyr-dcy)*dfzr - 2*(dpzr-dcz)*dfyr
        
        # iam is the integral of the angular momentum approximated by the base orientation.
        # am dam ddam and dddam are the angular momentum derivatives 
        
        # take the same gains as the CoM
        K_iam, K_am, K_dam, K_ddam  =  self.Kp_com, self.Kd_com, self.Ka_com, self.Kj_com
        iam_ref, am_ref, dam_ref, ddam_ref, dddam_ref = 0.,0.,0.,0.,0.
        Jam = self.robot.get_angularMomentumJacobian(q,v)
        robotInertia = Jam[0,2] 
        
        # measurements
        theta = np.arctan2(q[3],q[2])
        iam   = robotInertia * theta
        am    = am_est 
        dam   = (pyl-cy)*fzl                  -  (pzl-cz)*fyl                  + (pyr-cy)*fzr                 -  (pzr-cz)*fyr
        ddam  = (dpyl-dcy)*fzl+(pyl-cy)*dfzl - ((dpzl-dcz)*fyl+(pzl-cz)*dfyl) + (dpyr-dcy)*fzr+(pyr-cy)*dfzr - ((dpzr-dcz)*fyr+(pzr-cz)*dfyr)

        iam_err   =  iam -  iam_ref
        am_err    =   am -   am_ref
        dam_err   =  dam -  dam_ref
        ddam_err  = ddam - ddam_ref
        dddam_des = -K_iam*iam_err -K_am*am_err -K_dam*dam_err -K_ddam*ddam_err +dddam_ref
        
        self.data.robotInertia = robotInertia        
        self.data.iam  = iam
        self.data.am   = am
        self.data.dam  = dam
        self.data.ddam = ddam
        self.data.dddam_des = dddam_des
        
        return dddam_des, Xl, Nl
        
        
    def solve(self, t, q, v, f_meas, df_meas=None):
        robot=self.robot
        NQ,NV,NB,RF,LF,RK,LK = self.NQ,self.NV,self.NB,self.RF,self.LF,self.RK,self.LK
        w_post = self.w_post
        Kp_post, Kd_post = self.Kp_post, self.Kd_post

        se3.computeAllTerms(robot.model,robot.data,q,v)
        se3.updateFramePlacements(robot.model,robot.data)
        se3.rnea(robot.model,robot.data,q,v,0*v)
        M = robot.data.M        #(7,7)
        h = robot.data.nle      #(7,1)
        Jl,Jr = robot.get_Jl_Jr_world(q, False)
        Mlf, Mrf = robot.get_Mlf_Mrf(q, False)
        
        #4th order CoM via feet acceleration task **********************
        '''com_s_des -> ddf_des -> feet_a_des'''        
        
        # Estimate center of mass and derivatives
        if self.estimator is None:
            #take the measurement state as estimate, assume jerk is measured by df
            com_mes, com_v_mes, com_a_mes, com_j_mes = robot.get_com_and_derivatives(q, v, f_meas, df_meas)
            am_est = robot.get_angularMomentum(q, v)
            com_est, com_v_est, com_a_est, com_j_est, f_est, df_est = com_mes, com_v_mes, com_a_mes, com_j_mes, f_meas, df_meas
        else:
            com_mes, com_v_mes, com_a_mes = robot.get_com_and_derivatives(q, v, f_meas)
            am = robot.get_angularMomentum(q, v)
            p = np.hstack((Mlf.translation[1:].A1, Mrf.translation[1:].A1))
            self.estimator.predict_update(com_mes.A1, com_v_mes.A1, np.array([am]), f_meas.A1, p, self.data.ddf_des.A1)
            
            (com_est, com_v_est, am_est, f_est, df_est) = self.estimator.get_state(True)
            dummy1, dummy2, com_a_est, com_j_est = robot.get_com_and_derivatives(q, v, f_est, df_est)
            
            # DEBUG
#            com_mes, com_v_mes, com_a_mes, com_j_mes = robot.get_com_and_derivatives(q, v, f_meas, df_meas)
#            am_est = robot.get_angularMomentum(q, v)
#            com_est, com_v_est, com_a_est, com_j_est = com_mes, com_v_mes, com_a_mes, com_j_mes
#            f_est, df_est = f_meas, df_meas
#            f_est = f_meas
        
        
        # Formulate dynamic constrains  (ne sert a rien...)************* todo remove and make the problem smaller
        #      |M   -Jc.T  -S.T| * [dv,f,tau].T =  -h
        Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
        driftLF,driftRF = robot.get_driftLF_driftRF_world(q,v, False)
        dJcdq = np.vstack([driftLF.vector[1:3],driftRF.vector[1:3]])
        
        S  = np.hstack([matlib.zeros([4,3]), matlib.eye(4)]) # (4,7)
        Aec = np.vstack([np.hstack([M ,-Jc.T,-S.T]),
                         np.hstack([matlib.zeros([4,7]), matlib.eye(4), matlib.zeros([4,4])])])
#        bec = np.vstack([-h, f_est])
#        bec = np.vstack([-h, f_meas])
        # use f+dt/2*df instead of f
        bec = np.vstack([-h, f_meas + 0.5*self.dt*df_est])
        
 
        # CoM TASK COMPUTATIONS
        com_s_des, Xc, Nc = self._compute_com_task(t, com_est, com_v_est, com_a_est, com_j_est)
        
        # ANGULAR MOMENTUM TASK COMPUTATIONS
        dddam_des, Xl, Nl = self._compute_ang_mom_task(q, v, am_est, com_est, com_v_est, com_a_est, f_est, df_est)
        
        u_des = np.vstack([com_s_des, dddam_des]) # CoM + Angular Momentum
        X     = np.vstack([Xc, Xl])               # CoM + Angular Momentum
        N     = np.vstack([Nc, Nl])               # CoM + Angular Momentum
        
        b_momentum = u_des - N - X*self.Kspring*dJcdq
        A_momentum = np.hstack([X*self.Kspring*Jc, matlib.zeros([X.shape[0],4+4])]) # zeros for columns corresponding to forces and torques

        #posture task  *************************************************
        post_p_err     = matlib.zeros(7).T
        post_p_err[:2] = q[:2] - self.post_p_ref[:2] # Y-Z error
        post_p_err[2]  = np.arcsin(q[3]) - np.arcsin(self.post_p_ref[3]) # Angular error
        post_p_err[3:] = q[4:] - self.post_p_ref[4:] # Actuation error
        post_v_err     = v - self.post_v_ref
        post_a_des     = self.post_a_ref - Kp_post*post_p_err - Kd_post*post_v_err
        b_post         = w_post*post_a_des   

        #stack all tasks
        A=np.vstack([A_momentum, self.A_post])
        b=np.vstack([b_momentum, b_post])
        
        # Friction cone inequality constraints Aic * x >= bic
        dt_f = 1.1*self.dt
        
#        Aic = -dt_f*B_f*self.Kspring*Jc
#        viab_marg = -B_f*df_est + np.sqrt(-2*self.ddf_max*(B_f*f_est-b_f))
#        if (viab_marg<=0.0).any():
#            print "%.3f Warning: negative viability margins!"%(t), viab_marg.T
#            
#        margin = b_f - B_f*(f_est+dt_f*df_est)
#        if (margin<=0.0).any():
#            margin[margin<=0.0] = 0.0
#            print "%.3f Warning: negative predicted margins!"%(t), margin.T
#        bic = -np.sqrt(2*self.ddf_max*margin) + B_f*df_est + dt_f*B_f*self.Kspring*dJcdq
        
        b1 = np.matlib.ones(self.b_f.shape[0]).T
        B_ddf_max = np.abs(self.B_f*np.matlib.ones((4,1))*self.ddf_max)
        pos = self.B_f*f_est
        vel = self.B_f*df_est
        pos_min = -1e100*b1
        pos_max = self.b_f.copy()
        vel_max = 1e100*b1
        
#        viabViol = areStatesViable(pos, vel, pos_min, pos_max, vel_max, B_ddf_max);
#        if(np.sum(viabViol)>0):
#            print "Some constraints are not viable", np.where(viabViol)[0]
        
        ind_q = (pos>=pos_max).A1
        if(np.sum(ind_q)>0):
            new_pos_max = compute_min_q_upper_bound(pos[ind_q], vel[ind_q], B_ddf_max[ind_q])
#            print "TSID-FLEX WARNING: some forces violate friction cones:", np.where(ind_q)[0], (pos[ind_q]-pos_max[ind_q]).T,
#            print " old UB", pos_max[ind_q].T, "new UB", new_pos_max.T
            print "TSID-FLEX WARNING: update B_f_max", np.where(ind_q)[0], "from", pos_max[ind_q].T, "to", new_pos_max.T
            pos_max[ind_q] = new_pos_max + 1e-6
#            pos_max[ind_q] = pos[ind_q] + 0.1
            
        B_ddf_stop = compute_min_ddq_stop(pos, vel, pos_min, pos_max);
        ind = (B_ddf_stop > B_ddf_max).A1
        if(np.sum(ind)>0):
            self.ind = ind
            print "TSID-FLEX WARNING: update B_ddf_max", np.where(ind)[0], "from", B_ddf_max[ind].T, "to", B_ddf_stop[ind].T
            B_ddf_max[ind] = B_ddf_stop[ind]
            
        (B_ddf_LB, B_ddf_UB) = computeAccLimits(pos, vel, pos_min, pos_max, vel_max, B_ddf_max, dt_f, verbose=True, IMPOSE_ACCELERATION_BOUNDS=False)
            
        # B * ddf <= B_ddf_max
        # B * K * ddp <= B_ddf_max
        # B * K * J * dv <= B_ddf_max - B * K * dJdq
        # -B * K * J * dv >= -B_ddf_max + B * K * dJdq
        Aic = -self.B_f*self.Kspring*Jc
        bic = -B_ddf_UB + self.B_f*self.Kspring*dJcdq
        
        Aic = np.hstack([Aic , matlib.zeros((Aic.shape[0], 8))])
        
        #stack equality and inequality constrains
        Ac = np.vstack([Aec,Aic])
        bc = np.vstack([bec,bic])
#        Ac=Aec
#        bc=bec
        #formulate the least squares as a quadratic problem *************
        H=(A.T*A).T + self.HESSIAN_REGULARIZATION*np.eye(A.shape[1])
        g=(A.T*b).T
        y_QP = solve_qp(H.A, g.A1, Ac.A.T, bc.A1, bec.shape[0])[0]
        
        # compute active inequalities
        margins = Aic*np.matrix(y_QP).T - bic
        if (margins<=1e-5).any():
#            print "%.4f Active inequalities:"%(t), np.where(margins<=1e-5)[0] #margins.T
            self.ind = (margins<=1e-5).A1
        
        # Solve with inverse *******************************************
        if 0:        
            Mu = robot.data.M[:3]
            JuT = Jc.T[:3]
            hu = h[:3]
            A_ = np.vstack([ Mu,Jc,w_post*Jpost ])
            b_ = np.vstack([ JuT*fc - hu, feet_a_des - dJcdq, w_post*post_a_des  ])
            #~ A_ = np.vstack([ Mu,Jc])
            #~ b_ = np.vstack([ JuT*fc - hu, feet_a_des - dJcdq ])              
            dv_PINV = np.linalg.pinv(A_)*b_
            tau_PINV = S*(M*dv_PINV+h-Jc.T*fc)  #get tau_des from dv_des and measured contact forces:
            f_PINV=fc
            y_PINV = np.vstack([dv_PINV,f_PINV,tau_PINV]).A1
            assert isapprox(y_PINV,y_QP)

        dv  = np.matrix(y_QP[:7]   ).T
        f   = np.matrix(y_QP[7:7+4]).T
        tau = np.matrix(y_QP[7+4:] ).T
                
        # compute ddf_des for next loop EKF computations
        feet_a_des = Jc*dv + dJcdq
        self.data.ddf_des = self.Kspring * feet_a_des
        
        # store data
        self.data.lf_a_des = feet_a_des[:2]
        self.data.rf_a_des = feet_a_des[2:]
        
        self.data.com_p_mes  = com_mes.A1
        self.data.com_v_mes  = com_v_mes.A1
        self.data.com_a_mes  = com_a_mes.A1
        #~ self.data.com_j_mes  = com_j_mes.A1 # should not be able to measure jerk
        
        self.data.com_p_est  = com_est.A1
        self.data.com_v_est  = com_v_est.A1
        self.data.com_a_est  = com_a_est.A1
        self.data.com_j_est  = com_j_est.A1        
        self.data.com_s_des  = com_s_des.A1
        self.data.com_s_exp  = (X*self.data.ddf_des).A1
        
        self.data.lkf    = f[:2].A1
        self.data.rkf    = f[2:].A1
        self.data.tau    = tau
        self.data.dv     = dv
        self.data.f      = f
        
        self.data.B_ddf_max = - B_ddf_max
        self.data.B_ddf_UB  = - B_ddf_UB
        self.data.B_ddf_des = - self.B_f*self.data.ddf_des
        self.data.B_df      = - vel
        (B_df_min, B_df_max) = computeVelViabBounds(pos, pos_min, pos_max, B_ddf_max);
        self.data.B_df_max  = - B_df_max
        self.data.B_f       = pos_max - pos
        
        DEBUG_VIAB = 0
        if DEBUG_VIAB:
            dt = self.dt
            pos_next = pos + dt*vel + 0.5*dt*dt*self.data.B_ddf_des
            vel_next = vel + dt*self.data.B_ddf_des
            max_pos_next = pos + dt*vel + 0.5*dt*dt*B_ddf_UB
            max_vel_next = vel + dt*B_ddf_UB
            viabViol = areStatesViable(max_pos_next, max_vel_next, pos_min, pos_max, vel_max, B_ddf_max);
            if(np.sum(viabViol)>0):
                print "%.4f ERROR Some constraints will not be viable at next time step"%(t), np.where(viabViol)[0]
#            if(t>0.16):
#                print "    %.4f Current    value of B_f "%(t), pos.T
#                print "    %.4f Future     value of B_f "%(t), pos_next.T
#                print "    %.4f Max future value of B_f "%(t), max_pos_next.T
#                print "    %.4f Current    value of B_df"%(t), vel.T                
#                print "    %.4f Future     value of B_df"%(t), vel_next.T
#                print "    %.4f Max future value of B_df"%(t), max_vel_next.T
        
        if(np.sum(ind)>0):
            B_ddf_max = np.abs(self.B_f*np.matlib.ones((4,1))*self.ddf_max)
#            print "%.4f WARNING: increase max constr acc"%(t), np.where(ind)[0], 'from', B_ddf_max[ind].T, 'to', B_ddf_stop[ind].T, \
#                    'B_f', (pos_max[ind]-pos[ind]).T
#            print "%.4f WARNING: increase max constr acc"%(t), np.where(ind)[0], 'from', B_ddf_max[ind].T, 'to', B_ddf_stop[ind].T, \
#                    'B_ddf_UB', B_ddf_UB[ind].T, 'B_ddf_des', self.data.B_ddf_des[ind].T, 'B_df', vel[ind].T, 'B_f', (pos_max[ind]-pos[ind]).T
#        elif(np.sum(self.ind)>0):
#            ind = self.ind
#            print "%.4f DEBUG:              "%(t), np.where(ind)[0], 'B_ddf_max', B_ddf_max[ind].T, 'B_ddf_stop', B_ddf_stop[ind].T, \
#                    'B_ddf_UB', B_ddf_UB[ind].T, 'B_ddf_des', self.data.B_ddf_des[ind].T, 'B_df', vel[ind].T, 'B_f', (pos_max[ind]-pos[ind]).T
                    

        if w_post == 0 and 0:
            #Test that feet acceleration are satisfied
            try:    
                assert(isapprox(feet_a_des , Jc*dv + dJcdq))
            except AssertionError:
                print("Solution violate the desired feet acceleration and so the CoM snap {}").format(np.linalg.norm(feet_a_des -( Jc*dv + dJcdq)))
            #external forces should be zero
            assert(isapprox((M*dv+h-Jc.T*fc)[:3] , np.zeros([3,1]))) 
        return np.vstack([zero(3),tau])
        
if __name__ == "__main__":
    '''benchmark TSID'''
    from hrp2_reduced import Hrp2Reduced
    import time
    from path import pkg, urdf 
    np.set_printoptions(precision=3)
    np.set_printoptions(linewidth=200)
    robot = Hrp2Reduced(urdf,[pkg],loadModel=True,useViewer=False)
    tsid=TsidFlexibleContact(robot,Ky=23770.,Kz=239018.,w_post=0e-3,Kp_post=1.,Kp_com=1., Kd_com=1., Ka_com=1., Kj_com=1.)
    niter= 3000
    t0 = time.time()
    for i in range(niter):
        tsid.solve(robot.q0,np.zeros([7,1]),np.matrix([0.,0.,0.,0.]).T,np.matrix([0.,0.,0.,0.]).T)
    print 'TSID average time = %.5f ms' % ((time.time()-t0)/(1e-3*niter)) #0.53ms
    
    try:
        embed()
    except:
        pass
