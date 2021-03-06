import pinocchio as se3
from pinocchio.utils import *
from math import pi,sqrt,cos,sin
from quadprog import solve_qp
import numpy as np
from numpy import matlib
from utils.tsid_utils import createContactForceInequalities

try:
    from IPython import embed
except ImportError:
    pass

class Empty:
    pass


class GainsAdmCtrl:
    
    def __init__(self, gain_array=None, nf=2):
        '''
            gain_array: numpy array containing all the gains
            nf:         number of force feedback gains
        '''
        self.nf = nf
        if gain_array is None:
            gain_array = np.zeros(4+nf)
        self.from_array(gain_array)
    
    def to_array(self):
        res = np.zeros(4+self.nf)
#        res[0] = self.Kp_adm
#        res[1] = self.Kd_adm
        res[0] = self.Kp_com
        res[1] = self.Kd_com
        res[2] = self.kp_bar    #    Kp_pos = kp_bar*Mj_diag
        res[3] = self.kd_bar    #    Kd_pos = kd_bar*Mj_diag
        res[4:4+self.nf] = np.diag(self.Kf)[:2]
        return res
        
    def from_array(self, gains):
#        self.Kp_adm = gains[0]
#        self.Kd_adm = gains[1]
        self.Kp_com = gains[0]
        self.Kd_com = gains[1]
        self.kp_bar = gains[2]
        self.kd_bar = gains[3]
        self.Kf     = np.asmatrix(np.diag(np.concatenate((gains[4:4+self.nf],gains[4:4+self.nf]))))
        
    def to_string(self):
        res = ''
        for s in ['Kp_com', 'Kd_com', 'kp_bar', 'kd_bar']:
            res += s+' = '+str(self.__dict__[s])+'\n'
        res += 'Kf = 1e-4*np.diag('+str([v for v in 1e4*np.diag(self.Kf)])+')'
        return res
        
    @staticmethod
    def get_default_gains(K):
        gains = GainsAdmCtrl()
        K_inv = 2e-5*matlib.eye(4) #np.linalg.inv(K)
        gains.Kp_com, gains.Kd_com, gains.Kf = 30.6694018561, 10.2970910213, 400*K_inv # poles 5, 15, 25, 35
        gains.kp_bar = 1e4
        gains.kd_bar = 200.0
        return gains

    
class AdmittanceControl:
    
    HESSIAN_REGULARIZATION = 1e-8
    NEGLECT_FRICTION_CONES = False
    
    def __init__(self, robot, dt, q0, Ky, Kz, w_post, Kp_post, gains, fMin=0.0, mu=0.3, estimator=None):
        # specifies whether inverse kinematics (IK) is computed at the level of joint velocities (vel) or accelerations (acc)
        self.ik_strategy = 'vel'
        # specifies whether inverse kinematics (IK) considers the whole body or each limb independently
        self.ik_whole_body = False
        # specifies whether the force error is used to compute the contact point velocities (vel) or positions (pos)
        self.admittance_strategy = 'vel'
        
        self.robot = robot
        self.dt = dt
        self.estimator = estimator
        self.NQ = robot.model.nq
        self.NV = robot.model.nv
        self.NB = robot.model.nbodies
        self.RF = robot.model.getFrameId('rankle')
        self.LF = robot.model.getFrameId('lankle')
        self.RK = robot.model.frames[self.RF].parent
        self.LK = robot.model.frames[self.LF].parent
        
        Kspring = -np.matrix(np.diagflat([Ky,Kz,Ky,Kz]))   # Stiffness of the feet spring
        self.Kinv = np.linalg.inv(Kspring)
        self.Kf     = gains.Kf #-self.Kinv
        # try to read Kp_adm and Kd_adm, but they may not be specified if not needed
        try: self.Kp_adm = gains.Kp_adm
        except: pass
        try: self.Kd_adm = gains.Kd_adm
        except: pass
        self.fMin = fMin
        self.mu = mu
        
        self.w_post = w_post
        self.Kp_post = Kp_post
        self.Kd_post = 2*sqrt(Kp_post)
        self.Kp_com = gains.Kp_com
        self.Kd_com = gains.Kd_com
        self.dq_cmd = matlib.zeros((self.NV-3,1))
        self.q_cmd  = q0[4:,0]
        self.Kp_pos = gains.Kp_pos
        self.Kd_pos = gains.Kd_pos
        
        self.data = Empty()
        com_p_ref = np.matrix([0.,0.53]).T
        com_v_ref = np.matrix([0.,0.]).T
        com_a_ref = np.matrix([0.,0.]).T
        com_j_ref = np.matrix([0.,0.]).T
        com_s_ref = np.matrix([0.,0.]).T
        self.callback_com = lambda t: (com_p_ref,com_v_ref,com_a_ref,com_j_ref,com_s_ref )
        
        self.g = robot.model.gravity.linear[1:]
        self.m = robot.data.mass[0]
        self.X_com = np.hstack([np.eye(2),np.eye(2)])
        self.p_0 = -self.Kinv*np.linalg.pinv(self.X_com)*self.m*self.g
        
            
    def solve(self, t, q, v, f_meas, df_meas=None):
        robot=self.robot
        NQ,NV,NB,RF,LF,RK,LK = self.NQ,self.NV,self.NB,self.RF,self.LF,self.RK,self.LK
        w_post = self.w_post
        Kp_post, Kd_post = self.Kp_post, self.Kd_post
        Kp_com,  Kd_com  = self.Kp_com,  self.Kd_com
        
        se3.computeAllTerms(robot.model, robot.data, q, v)
        se3.updateFramePlacements(robot.model, robot.data)
        se3.rnea(robot.model, robot.data, q, v, 0*v)
        Jl,Jr = robot.get_Jl_Jr_world(q, False)
        Mlf, Mrf = robot.get_Mlf_Mrf(q, False)
        
        if(self.estimator is None):
            com_mes, com_v_mes = robot.get_com_and_derivatives(q, v)
            am_est = robot.get_angularMomentum(q, v)
            com_est, com_v_est, f_est, df_est = com_mes, com_v_mes, f_meas, df_meas
        else:
            com_mes, com_v_mes, com_a_mes = robot.get_com_and_derivatives(q, v, f_meas)
            am = robot.get_angularMomentum(q, v)
            p = np.hstack((Mlf.translation[1:].A1, Mrf.translation[1:].A1))
            self.estimator.predict_update(com_mes.A1, com_v_mes.A1, np.array([am]), f_meas.A1, p)            
            (com_est, com_v_est, am_est, f_est, df_est) = self.estimator.get_state(True)

        Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
        driftLF,driftRF = robot.get_driftLF_driftRF_world(q,v, False)
        dJcdq = np.vstack([driftLF.vector[1:3],driftRF.vector[1:3]])
        
        #CoM task ******************************************************
        com_p_ref, com_v_ref, com_a_ref, com_j_ref, com_s_ref = self.callback_com(t)
        com_p_err =   com_est - com_p_ref
        com_v_err = com_v_est - com_v_ref
        com_a_des = -Kp_com*com_p_err -Kd_com*com_v_err +com_a_ref
        
        Jam = robot.get_angularMomentumJacobian(q,v)
        robotInertia = Jam[0,2] 
        
        # measurements
        Mlf, Mrf = robot.get_Mlf_Mrf(q, False) # This should come from the forces, which can be filtered, but need for p0l p0r
        pyl, pzl = Mlf.translation[1:].A1
        pyr, pzr = Mrf.translation[1:].A1
        fyl, fzl, fyr, fzr = f_est.A1
        cy, cz     = com_est.A1
        theta = np.arctan2(q[3],q[2])
        iam = robotInertia * theta
        am = am_est #(Jam*v).A1[0] # Jam*v 3d ???
        fyl, fzl, fyr, fzr = f_est.A1
        cy, cz     = com_est.A1
        X_am  = np.matrix([-(pzl-cz),+(pyl-cy),-(pzr-cz),+(pyr-cy)])
        
        Kp_am, Kd_am = Kp_com, Kd_com
        iam_ref, am_ref = 0.0, 0.0
        iam_err  =  iam -  iam_ref
        am_err   =   am -   am_ref
        dam_des = -Kp_am*iam_err -Kd_am*am_err         
                
        #com + am
        X = np.vstack([self.X_com, X_am])
        ddx_des = np.vstack([self.m*(com_a_des-self.g), dam_des])
        if(self.NEGLECT_FRICTION_CONES):
            X_pinv = np.linalg.pinv(X)
            f_des = X_pinv*ddx_des
        else:
            # Friction cone inequality constraints Aic * x >= bic
            if(self.fMin==0.0): k = 2
            else:               k = 3
            B_f = matlib.zeros((2*k,4));
            b_f = matlib.zeros(B_f.shape[0]).T;
            (B_f[:k,:2], b_f[:k,0]) = createContactForceInequalities(self.mu, self.fMin) # B_f * f <= b_f
            (B_f[k:,2:], b_f[k:,0]) = createContactForceInequalities(self.mu, self.fMin)
            
            H=(X.T*X).T + self.HESSIAN_REGULARIZATION*np.eye(X.shape[1])
            g=(X.T*ddx_des).T
            f_des = np.matrix(solve_qp(H.A, g.A1, -B_f.T.A, -b_f.A1, 0)[0]).T

        if(self.admittance_strategy=='pos'):
            feet_p_ref = self.p_0 - self.Kf*(f_des - f_est)
        elif(self.admittance_strategy=='vel'):
            feet_v_ref = - self.Kf*(f_des - f_est)

        A_admcom  = Jc
        if(not self.ik_whole_body):
            # set to zero the columns corresponding to floating-base
            A_admcom[:,:3] = 0.0
            
        if(self.ik_strategy=='vel'):            
            if(self.admittance_strategy=='pos'):
                feet_p = self.Kinv*f_est    # the minus sign is already included in Kinv
                feet_v_des = self.Kp_adm * (feet_p_ref - feet_p)
            elif(self.admittance_strategy=='vel'):
                feet_v_des = feet_v_ref
            b_admcom  = feet_v_des
        elif(self.ik_strategy=='acc'):
            feet_v = self.Kinv*df_est
            if(self.admittance_strategy=='pos'):
                feet_p = self.Kinv*f_est    # the minus sign is already included in Kinv                
                feet_a_des = self.Kp_adm * (feet_p_ref - feet_p) - self.Kd_adm * feet_v
            elif(self.admittance_strategy=='vel'):
                feet_a_des = self.Kd_adm * (feet_v_ref - feet_v)
            b_admcom  = feet_a_des - dJcdq
        
        #posture task  *************************************************
        post_p_ref = robot.q0[4:] #only the actuated part !
        post_v_ref = np.matrix([0,0,0,0]).T
        post_a_ref = np.matrix([0,0,0,0]).T
        post_p_err = q[4:] - post_p_ref
        post_v_err = v[3:] - post_v_ref
        A_post  = w_post*np.hstack([np.zeros([4,3]),np.eye(4)])
        if(self.ik_strategy=='vel'):            
            post_v_des = post_v_ref - Kp_post*post_p_err            
            b_post  = w_post*post_v_des
        elif(self.ik_strategy=='acc'):
            post_a_des = post_a_ref - Kp_post*post_p_err - Kd_post*post_v_err 
            b_post  = w_post*post_a_des
        
        #stack all tasks
        A = np.vstack([A_admcom, A_post])
        b = np.vstack([b_admcom, b_post])
                
        #formulate the least square as a quadratic problem *************
        H=(A.T*A).T + self.HESSIAN_REGULARIZATION*np.eye(A.shape[1])
        g=(A.T*b).T
        
        #solve it ******************************************************
        y = solve_qp(H.A, g.A1)[0] #, Ac.T.A, bc.A1, bec.shape[0])[0]
        
        if(self.ik_strategy=='vel'):            
            dq_cmd = np.matrix(y[3:]).T
            # integrate desired dq to get desired joint angles        
            self.q_cmd  += self.dt*dq_cmd
        elif(self.ik_strategy=='acc'):
            ddq_cmd = np.matrix(y[3:]).T        
            # integrate desired dv twice to get desired joint angles        
            self.q_cmd  += self.dt*self.dq_cmd + 0.5*(self.dt**2)*ddq_cmd
            self.dq_cmd += self.dt*ddq_cmd
            
        tau = self.Kp_pos*(self.q_cmd-q[4:]) - self.Kd_pos*v[3:]
                
        #populate results
        if(self.ik_strategy=='acc'):
            self.data.lf_a_des = feet_a_des[:2]
            self.data.rf_a_des = feet_a_des[2:]
            self.data.dv   = np.matrix(y).T
        elif(self.ik_strategy=='vel'):
            self.data.lf_v_des = feet_v_des[:2]
            self.data.rf_v_des = feet_v_des[2:]
            self.data.v_cmd    = np.matrix(y).T
            self.data.dv = matlib.zeros((self.robot.model.nv,1))
            
        self.data.com_p_mes  = com_mes.A1
        self.data.com_v_mes  = com_v_mes.A1
        self.data.com_a_mes  = np.matrix([0.,0.]).T.A1
        #~ self.data.com_j_mes  = com_j_mes.A1 # should not be able to measure jerk
        
        self.data.com_p_est  = com_est.A1
        self.data.com_v_est  = com_v_est.A1
        self.data.com_a_est  = np.matrix([0.,0.]).T.A1
        self.data.com_j_est  = np.matrix([0.,0.]).T.A1
        
        self.data.com_a_des = com_a_des.A1
        self.data.com_s_des = np.matrix([0.,0.]).T.A1 #None in rigid contact controller
        
        self.data.com_p_err = com_p_err.A1
        self.data.com_v_err = com_v_err.A1
        self.data.com_a_err = np.matrix([0.,0.]).T.A1
        self.data.com_j_err = np.matrix([0.,0.]).T.A1
        self.data.comref = com_p_ref.A1
        
        self.data.robotInertia = 0

        self.data.iam  = iam
        self.data.am   = am
        self.data.dam  = 0
        self.data.ddam = 0
        self.data.dddam_des = 0 #None in rigid contact controller
        
        self.data.lkf    = f_des[:2].A1 #f[:2].A1
        self.data.rkf    = f_des[2:].A1
        self.data.tau    = tau
        self.data.q_cmd  = self.q_cmd
        
        return np.vstack([zero(3),tau])
        

if __name__ == "__main__":
    '''benchmark TSID'''
    from hrp2_reduced import Hrp2Reduced
    from IPython import embed
    import time
    from path import pkg, urdf 
    robot = Hrp2Reduced(urdf,[pkg],loadModel=True,useViewer=False)
    tsid=Tsid(robot,100,100,Kp_post=10,Kp_com=30,w_post=0.001)
    niter= 3000
    t0 = time.time()
    for i in range(niter):
        tsid.solve(robot.q0,np.zeros([7,1]),np.matrix([0.,0.,0.,0.]).T,np.matrix([0.,0.,0.,0.]).T)
    print 'TSID average time = %.5f ms' % ((time.time()-t0)/(1e-3*niter)) #0.53ms
    embed()
