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
    
class TsidAdmittance:
    
    HESSIAN_REGULARIZATION = 1e-8
    NEGLECT_FRICTION_CONES = False
    
    def __init__(self, robot, Ky, Kz, w_post, Kp_post, Kp_com, Kf, Kp_adm=400.0, Kd_adm=40.0, fMin=0.0, mu=0.3, estimator=None):
        self.robot = robot
        self.estimator = estimator
        self.NQ = robot.model.nq
        self.NV = robot.model.nv
        self.NB = robot.model.nbodies
        self.RF = robot.model.getFrameId('rankle')
        self.LF = robot.model.getFrameId('lankle')
        self.RK = robot.model.frames[self.RF].parent
        self.LK = robot.model.frames[self.LF].parent
        
        self.Ky = Ky
        self.Kz = Kz
        self.Kspring = -np.matrix(np.diagflat([Ky,Kz,Ky,Kz]))   # Stiffness of the feet spring
        self.Kinv = np.linalg.inv(self.Kspring)
        self.Kf = Kf #-self.Kinv
        self.Kp_adm = Kp_adm
        self.Kd_adm = Kd_adm
        self.fMin = fMin
        self.mu = mu
        
        self.w_post = w_post
        self.Kp_post = Kp_post
        self.Kd_post = 2*sqrt(Kp_post)
        self.Kp_com = Kp_com
        self.Kd_com = 2*sqrt(Kp_com)
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
        se3.framesKinematics(robot.model, robot.data, q)
        se3.rnea(robot.model, robot.data, q, v, 0*v)
        m = self.m
        g = self.g
        M = robot.data.M        #(7,7)
        h = robot.data.nle      #(7,1)
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
            
            # DEBUG
#            f_est_err = np.abs(f_est-f_meas)
#            for i in range(f_est_err.size):
#                if(f_est_err[i,0]>self.f_est_err_max[i,0]):
#                    self.f_est_err_max[i,0] = f_est_err[i,0]
#                    print "Time %.3f   Max force est err %d: %.1f (%.1f)" % (t, i, f_est_err[i,0], 1e2*f_est_err[i,0]/f_meas[i,0])
            
            # use the measurements for the contact forces and the CoM position
#            com_est, f_est, df_est = com_mes, f_meas, df_meas
        
        # Formulate contact and dynamic constrains *********************
        #        7    4     4       7  4  4
        # 6   |Jc    0      0 | * [dv,f,tau].T =  |-dJc*dq|
        # 7   |M   -Jc.T  -S.T|                   |  -h   |

        Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
        S  = np.hstack([np.zeros([4,3]),np.eye(4)]) # (4,7)
        driftLF,driftRF = robot.get_driftLF_driftRF_world(q,v, False)
        dJcdq = np.vstack([driftLF.vector[1:3],driftRF.vector[1:3]])
        
        #friction cone constrains
#        mu = 0.5 # need more realistic value !!!!!!!!!!!!!!!!!!!!!!!!!!!
#        Aic = np.zeros([4,15])
#        bic = np.zeros([4,1])
#
#        # mu*Fz+Fy >0
#        Aic[0, 8] = mu  
#        Aic[0, 7] = 1.0
#        # mu*Fz-Fy >0
#        Aic[1,8] = mu  
#        Aic[1,7] = -1.0
#        # mu*Fz+Fy >0
#        Aic[2, 10] = mu  
#        Aic[2, 9] = 1.0
#        # mu*Fz-Fy >0
#        Aic[3,10] = mu  
#        Aic[3,9] = -1.0
        
        #CoM task ******************************************************
        com_p_ref, com_v_ref, com_a_ref, com_j_ref, com_s_ref = self.callback_com(t)
        com_p_err =   com_est - com_p_ref
        com_v_err = com_v_est - com_v_ref
        com_a_des = -Kp_com*com_p_err -Kd_com*com_v_err +com_a_ref


        #CoM task via ADMITANCE CONTROL ********************************
        Aec = np.vstack([np.hstack([M ,-Jc.T,-S.T]),
                         np.hstack([np.zeros([4,7]) ,np.eye(4),np.zeros([4,4])])])
        # Here it works better to use the measured forces rather than the estimated ones
        bec = np.vstack([-h, f_meas]) #f_est])
#        bec = np.vstack([-h, f_est])
        
        feet_p = self.Kinv*f_est    # the minus sign is already included in Kinv
        feet_v = self.Kinv*df_est
        
        Jam = robot.get_angularMomentumJacobian(q,v)
        robotInertia = Jam[0,2] 
        
        #measurments
        Mlf, Mrf = robot.get_Mlf_Mrf(q, False) # This should come from the forces wicth can be filtered, but need for p0l p0r
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
        ddx_des = np.vstack([m*(com_a_des-g), dam_des])
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
            
            # compute active inequalities
            margins = -B_f*f_des + b_f
#            if (margins<=1e-5).any():
#                print "%.3f Active inequalities:"%(t), np.where(margins<=1e-5)[0] #margins.T
#                print "f_des", f_des.T

        feet_p_des = self.p_0 - self.Kf*(f_des - f_est)
        feet_a_des = self.Kp_adm * (feet_p_des - feet_p) - self.Kd_adm * feet_v
        
        A_admcom  = np.hstack([Jc, np.matrix(np.zeros([4,8]))])
        b_admcom  = feet_a_des - dJcdq
        
        #posture task  *************************************************
        Jpost = np.hstack([np.zeros([4,3]),np.eye(4)])
        post_p_ref = robot.q0[4:] #only the actuated part !
        post_v_ref = np.matrix([0,0,0,0]).T
        post_a_ref = np.matrix([0,0,0,0]).T
        post_p_err = q[4:] - post_p_ref
        post_v_err = v[3:] - post_v_ref
        post_a_des = post_a_ref - Kp_post*post_p_err - Kd_post*post_v_err 

        z4 = np.matrix(np.zeros([4,4+4]))
        A_post  = w_post*np.hstack([Jpost,z4])
        b_post  = w_post*post_a_des # -post_drift
        
        #stack all tasks
        A=np.vstack([A_admcom,A_post])
        b=np.vstack([b_admcom,b_post])
        
        #stack equality and inequality constrains
#        Ac = np.vstack([Aec,Aic])
#        bc = np.vstack([bec,bic])
        Ac=Aec
        bc=bec
        
        #formulate the least square as a quadratic problem *************
        H=(A.T*A).T + self.HESSIAN_REGULARIZATION*np.eye(A.shape[1])
        g=(A.T*b).T
        
        #solve it ******************************************************
        y = solve_qp(H.A, g.A1, Ac.T.A, bc.A1, bec.shape[0])[0]
        #~ y = solve_qp(H,g)[0] # without constrains !
        #~ y = np.squeeze(np.asarray(np.linalg.pinv(A)*b)) # without constrains with pinv!

        dv = np.matrix(y[:7]   ).T
        f   = np.matrix(y[7:7+4]).T
        tau = np.matrix(y[7+4:] ).T
                
        #populate results
        #~ self.data.lf_a_des = np.matrix([0.,0.]).T 
        #~ self.data.rf_a_des = np.matrix([0.,0.]).T
        self.data.lf_a_des = feet_a_des[:2]
        self.data.rf_a_des = feet_a_des[2:]
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
        self.data.dv     = dv
        self.data.f      = f
        #~ print "feet_a_des"
        #~ print feet_a_des
        #~ print "Jc*dv + dJcdq"
        #~ print Jc*dv + dJcdq
        #~ assert(isapprox(feet_a_des , Jc*dv + dJcdq))
        #~ embed()
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
