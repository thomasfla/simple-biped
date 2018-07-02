import pinocchio as se3
from pinocchio.utils import *
from math import pi,sqrt,cos,sin
from quadprog import solve_qp
import matplotlib.pyplot as plt
from numpy import matlib

try:
    from IPython import embed
except ImportError:
    pass

class Empty:
    pass
    
class TsidMistry:
    HESSIAN_REGULARIZATION = 1e-8
    
    def __init__(self, robot, Ky, Kz, By, Bz, w_post, Kp_post, Kp_com, Kd_com, dt, estimator = None):
        self.robot = robot
        self.dt = dt
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
        self.Kp_com = Kp_com
        self.Kd_com = Kd_com
        self.Ky = Ky
        self.Kz = Kz
        self.By = By
        self.Bz = Bz
        self.ddf_des = matlib.zeros(4)
        self.data = Empty()
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
        self.Kspring    = -np.matrix(np.diagflat([Ky,Kz,Ky,Kz]))     # Stiffness of the feet spring
        self.Bspring    = -np.matrix(np.diagflat([By, Bz, By, Bz]))  # Damping of the feet spring
        self.Kinv       = np.linalg.inv(self.Kspring)
        self.g = robot.model.gravity.linear[1:]
        self.m = robot.data.mass[0]
     
    def _compute_com_task(self, t, com_est, com_v_est):
        Kp_com,  Kd_com = self.Kp_com, self.Kd_com
        com_p_ref, com_v_ref, com_a_ref, com_j_ref, com_s_ref = self.callback_com(t)
        com_p_err = com_est   - com_p_ref
        com_v_err = com_v_est - com_v_ref
        com_a_des = com_a_ref -Kp_com*com_p_err -Kd_com*com_v_err

        # Xc = (1./self.m) * np.hstack([np.eye(2),np.eye(2)])
        # Nc = self.g #np.matrix([0.,0.]).T
        
        self.data.com_p_err = com_p_err.A1
        self.data.com_v_err = com_v_err.A1
        self.data.comref    = com_p_ref.A1
        
        return com_a_des #, Xc, Nc
        
    def _compute_ang_mom_task(self, q, v, am_est, com_est):
        # Mlf, Mrf = self.robot.get_Mlf_Mrf(q, False)
        # pyl, pzl = Mlf.translation[1:].A1
        # pyr, pzr = Mrf.translation[1:].A1
        # cy, cz     = com_est.A1
        # Xl = np.matrix([ -(pzl - cz), (pyl - cy), -(pzr - cz), (pyr - cy)])
        # Nl = np.matrix([0.])

        # iam is the integral of the angular momentum approximated by the base orientation.
        # am dam ddam and dddam are the angular momentum derivatives 
        
        # take the same gains as the CoM
        K_iam, K_am =  self.Kp_com, self.Kd_com
        iam_ref, am_ref, dam_ref = 0.,0.,0.
        Jam = self.robot.get_angularMomentumJacobian(q,v)
        robotInertia = Jam[0,2] 
        
        # measurements
        theta = np.arctan2(q[3],q[2])
        iam   = robotInertia * theta
        am    = am_est
        iam_err   =  iam -  iam_ref
        am_err    =   am -   am_ref
        dam_des = -K_iam*iam_err -K_am*am_err + dam_ref
        
        self.data.robotInertia = robotInertia        
        self.data.iam  = iam
        self.data.am   = am
        self.data.dam_des = dam_des
        
        return dam_des #, Xl, Nl

    def _compute_Apc(self, q, com_est):
        Mlf, Mrf = self.robot.get_Mlf_Mrf(q, False)
        pyl, pzl = Mlf.translation[1:].A1
        pyr, pzr = Mrf.translation[1:].A1
        cy, cz = com_est.A1
        A_pc = matlib.zeros((3,4))
        A_pc[:2, :2] = matlib.eye(2)
        A_pc[:2, 2:] = matlib.eye(2)
        A_pc[2, :]   = np.matrix([-(pzl-cz), pyl-cy, -(pzr-cz), pyr-cy])
        b_g = np.vstack((self.m*self.g, matlib.zeros(1)))
        return A_pc, b_g

    def _compute_dApc(self, q, v, com_est, com_vel):
        # Mlf, Mrf = self.robot.get_Mlf_Mrf(q, False)
        # pyl, pzl = Mlf.translation[1:].A1
        # pyr, pzr = Mrf.translation[1:].A1
        vl, vr = self.robot.get_vlf_vrf_world(q, v, False)
        cy, cz = com_est.A1
        dcy, dcz = com_vel.A1
        dpyl, dpzl = vl.linear[1:].A1
        dpyr, dpzr = vr.linear[1:].A1
        dA_pc = matlib.zeros((3,4))
        dA_pc[2, :] = np.matrix([-(dpzl-dcz), dpyl-dcy, -(dpzr-dcz), dpyr-dcy])
        return dA_pc
        
    def solve(self, t, q, v, f_meas, df_meas=None):
        robot=self.robot
        NQ,NV,NB,RF,LF,RK,LK = self.NQ,self.NV,self.NB,self.RF,self.LF,self.RK,self.LK
        w_post = self.w_post
        Kp_post, Kd_post = self.Kp_post, self.Kd_post
        K, B = self.Kspring, self.Bspring
        m = self.m

        se3.computeAllTerms(robot.model,robot.data,q,v)
        se3.framesKinematics(robot.model,robot.data,q)
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
            self.estimator.predict_update(com_mes.A1, com_v_mes.A1, np.array([am]), f_meas.A1, p)
            
            (com_est, com_v_est, am_est, f_est, df_est) = self.estimator.get_state(True)
            dummy1, dummy2, com_a_est, com_j_est = robot.get_com_and_derivatives(q, v, f_est, df_est)
            
            # DEBUG
#            com_mes, com_v_mes, com_a_mes, com_j_mes = robot.get_com_and_derivatives(q, v, f_meas, df_meas)
#            am_est = robot.get_angularMomentum(q, v)
#            com_est, com_v_est, com_a_est, com_j_est = com_mes, com_v_mes, com_a_mes, com_j_mes
#            f_est = f_meas
#            df_est = df_meas
        
        
        # Formulate dynamic constrains ************* todo remove and make the problem smaller
        #      |M   -Jc.T  -S.T| * [dv,f,tau].T =  -h
        Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
        driftLF,driftRF = robot.get_driftLF_driftRF_world(q,v, False)
        dJcdq = np.vstack([driftLF.vector[1:3],driftRF.vector[1:3]])
        
        S  = np.hstack([matlib.zeros([4,3]), matlib.eye(4)]) # (4,7)
        Aec = np.vstack([np.hstack([M ,-Jc.T,-S.T]),
                         np.hstack([matlib.zeros([4,7]), matlib.eye(4), matlib.zeros([4,4])])])
        bec = np.vstack([-h, f_est])
        
 
        # CoM TASK COMPUTATIONS
        com_a_des = self._compute_com_task(t, com_est, com_v_est)
        # ANGULAR MOMENTUM TASK COMPUTATIONS
        dam_des = self._compute_ang_mom_task(q, v, am_est, com_est)
        
        ddx_des = np.vstack([m*com_a_des, dam_des])   # CoM + Angular Momentum
        A_pc, b_g  = self._compute_Apc(q, com_est)
        dA_pc = self._compute_dApc(q, v, com_est, com_v_est)
        A_dt  = A_pc + self.dt*dA_pc
        dp    = Jc*v # self.Kinv * df_est
        N     = A_dt*(f_est + self.dt*K*dp) + b_g - ddx_des
        X     = -self.dt*A_dt*B

        b_momentum = N - X*dJcdq
        A_momentum = np.hstack([X*Jc, matlib.zeros([X.shape[0],4+4])]) # zeros for columns corresponding to forces and torques

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
        
        #stack equality and inequality constrains
        #~ Ac = np.vstack([Aec,Aic])
        #~ bc = np.vstack([bec,bic])
        Ac=Aec
        bc=bec
        #formulate the least squares as a quadratic problem *************
        H=(A.T*A).T + self.HESSIAN_REGULARIZATION*np.eye(A.shape[1])
        g=(A.T*b).T
        try:
            y_QP = solve_qp(H.A, g.A1, Ac.A.T, bc.A1, bec.shape[0])[0]
        except:
            print t, "Error while solving QP. Singular values of Hessian:", np.linalg.svd(H, compute_uv=0)
            raise

        dv  = np.matrix(y_QP[:7]   ).T
        f   = np.matrix(y_QP[7:7+4]).T
        tau = np.matrix(y_QP[7+4:] ).T
                
        feet_a_des = Jc*dv + dJcdq
        self.df_des = K*Jc*v + B * feet_a_des
        
        # store data
        self.data.lf_a_des = feet_a_des[:2]
        self.data.rf_a_des = feet_a_des[2:]
        
        self.data.com_p_mes  = com_mes.A1
        self.data.com_v_mes  = com_v_mes.A1
        self.data.com_a_mes  = com_a_mes.A1
        
        self.data.com_p_est  = com_est.A1
        self.data.com_v_est  = com_v_est.A1
        self.data.com_a_est  = com_a_est.A1
        self.data.com_a_des  = com_a_des.A1
        
        self.data.lkf    = f[:2].A1
        self.data.rkf    = f[2:].A1
        self.data.tau    = tau
        self.data.dv     = dv
        self.data.f      = f

        return np.vstack([zero(3),tau])
