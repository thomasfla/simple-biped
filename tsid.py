import pinocchio as se3
from pinocchio.utils import *
from math import pi,sqrt,cos,sin
from quadprog import solve_qp
import matplotlib.pyplot as plt

try:
    from IPython import embed
except ImportError:
    pass

class Empty:
    pass
    
class Tsid:
    def __init__(self,robot,Ky,Kz,Kp_post,Kp_com,w_post):
        self.robot = robot
        self.NQ = robot.model.nq
        self.NV = robot.model.nv
        self.NB = robot.model.nbodies
        self.RF = robot.model.getFrameId('rankle')
        self.LF = robot.model.getFrameId('lankle')
        self.RK = robot.model.frames[self.RF].parent
        self.LK = robot.model.frames[self.LF].parent
        
        self.Ky = Ky
        self.Kz = Kz
        
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
        
    def setGains(self,P,V=None):
        if V is None: 
            V=2*sqrt(P)
            self.P,self.V = P,V
            
    def solve(self,q,v,fc,dfc,t=0.0):
        robot=self.robot
        NQ,NV,NB,RF,LF,RK,LK = self.NQ,self.NV,self.NB,self.RF,self.LF,self.RK,self.LK
        w_post = self.w_post
        Ky = self.Ky
        Kz = self.Kz
        Kp_post, Kd_post = self.Kp_post, self.Kd_post
        Kp_com,  Kd_com  = self.Kp_com,  self.Kd_com
        callback_com = self.callback_com

        se3.computeAllTerms(robot.model,robot.data,q,v)
        se3.framesKinematics(robot.model,robot.data,q)
        se3.rnea(robot.model,robot.data,q,v,0*v)
        m = robot.data.mass[0]
        g = robot.model.gravity.linear[1:]
        M  = robot.data.M        #(7,7)
        h  = robot.data.nle      #(7,1)
        
        Jl,Jr = robot.get_Jl_Jr_world(q, True)

        # Formulate contact and dynamic constrains *********************
        #        7    4     4       7  4  4
        # 6   |Jc    0      0 | * [dv,f,tau].T =  |-dJc*dq|
        # 7   |M   -Jc.T  -S.T|                   |  -h   |

        Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
        S  = np.hstack([np.zeros([4,3]),np.eye(4)]) # (4,7)
        z1 = np.matrix(np.zeros([4,4]))
        z2 = np.matrix(np.zeros([4,4]))

        driftLF,driftRF = robot.get_driftLF_driftRF_world(q,v, False)
        dJcdq = np.vstack([driftLF.vector[1:3],driftRF.vector[1:3]])

        Aec = np.vstack([ np.hstack([Jc, z1  ,z2])  ,
                          np.hstack([M ,-Jc.T,-S.T])  ]) 
        bec = np.vstack([-dJcdq,-h])
        #only contacts
        #~ Ac = np.hstack([Jc, z1  ,z2]) 
        #~ bc = -dJcdq 
        
        #only dynamics
        #~ Ac = np.hstack([M ,-Jc.T,-S.T])
        #~ bc = -h 

        
        #friction cone constrains
        mu = 0.5 # need more realistic value !!!!!!!!!!!!!!!!!!!!!!!!!!!
        Aic = np.zeros([4,15])
        bic = np.zeros([4,1])

        # mu*Fz+Fy >0
        Aic[0, 8] = mu  
        Aic[0, 7] = 1.0
        # mu*Fz-Fy >0
        Aic[1,8] = mu  
        Aic[1,7] = -1.0
        # mu*Fz+Fy >0
        Aic[2, 10] = mu  
        Aic[2, 9] = 1.0
        # mu*Fz-Fy >0
        Aic[3,10] = mu  
        Aic[3,9] = -1.0
        
        #CoM task ******************************************************
        Jcom = robot.data.Jcom[1:] #Only Y and Z

        com_p_ref, com_v_ref, com_a_ref, com_j_ref, com_s_ref = callback_com(t)
        se3.centerOfMass(robot.model,robot.data,q,v,zero(NV))
        com_mes   = robot.data.com[0][1:] 
        com_v_mes = robot.data.vcom[0][1:]
        
        com_est   = com_mes#Todo use an estimator
        com_v_est = com_v_mes
        
        com_drift = robot.data.acom[0][1:]
        com_p_err =   com_est - com_p_ref
        com_v_err = com_v_est - com_v_ref

        com_a_des = -Kp_com*com_p_err -Kd_com*com_v_err +com_a_ref

        z3 = np.matrix(np.zeros([2,4+4]))

        A_com  = np.hstack([Jcom,z3])
        b_com  = com_a_des - com_drift

        #CoM task via ADMITANCE CONTROL ********************************
        Aec = np.vstack([np.hstack([M ,-Jc.T,-S.T]),
                         np.hstack([np.zeros([4,7]) ,np.eye(4),np.zeros([4,4])])])
        bec = np.vstack([-h,
                         fc])
        
        Kspring = -np.matrix(np.diagflat([Ky,Kz,Ky,Kz]))   # Stiffness of the feet spring
        Kinv = np.linalg.inv(Kspring)
        
        #~ Mlf, Mrf = robot.get_Mlf_Mrf(q, False) 
        #~ pyl, pzl = Mlf.translation[1:].A1
        #~ pyr, pzr = Mrf.translation[1:].A1
        #~ feet_p = np.matrix([pyl,pzl,pyr, pzr]).T 
        feet_p = Kinv*fc
        feet_v = Kinv*dfc
        X_com = np.hstack([np.eye(2),np.eye(2)])
        
        Jam = robot.get_angularMomentumJacobian(q,v)
        robotInertia = Jam[0,2] 
        
        #measurments
        Mlf, Mrf = robot.get_Mlf_Mrf(q, False) # This should come from the forces wicth can be filtered, but need for p0l p0r
        pyl, pzl = Mlf.translation[1:].A1
        pyr, pzr = Mrf.translation[1:].A1
        fyl, fzl, fyr, fzr = fc.A1
        cy, cz     = com_est.A1
        theta = np.arctan2(q[3],q[2])
        iam = robotInertia * theta
        am = (Jam*v).A1[0] # Jam*v 3d ???
        fyl, fzl, fyr, fzr = fc.A1
        cy, cz     = com_est.A1
        X_am  = np.matrix([-(pzl-cz),+(pyl-cy),-(pzr-cz),+(pyr-cy)])
        
        Kp_am = Kp_com
        Kd_am = Kd_com
    
        iam_ref = 0
        am_ref  = 0
        
        iam_err  =  iam -  iam_ref
        am_err   =   am -   am_ref

        dam_des = -Kp_am*iam_err -Kd_am*am_err 
        
        
        Kf = Kinv 
        Kp = 400.
        Kd = 2*sqrt(Kp)
        
        #com only
        #~ X = X_com
        #~ X_pinv = np.linalg.pinv(X)
        #~ f_des = X_pinv*m*(com_a_des-g)
        
        
        #com + am
        X = np.vstack([X_com,
                       X_am])
        X_pinv = np.linalg.pinv(X)
        f_des = X_pinv*np.vstack( [m*(com_a_des-g),
                                       dam_des   ] )
                                       
                                       
        feet_p_des = -Kinv*np.linalg.pinv(X_com)*m*g + Kf*(f_des - fc)
        feet_a_des = Kp * (feet_p_des - feet_p) - Kd * feet_v
        #~ feet_a_des = np.matrix([0.,-10.,0.,-10]).T #TEST
        
        A_admcom  = np.hstack([Jc, np.matrix(np.zeros([4,8]))])
        b_admcom  = feet_a_des - dJcdq
        
        #posture task  *************************************************
        Jpost = np.hstack([np.zeros([4,3]),np.eye(4)])
        post_p_ref = robot.q0[4:] #only the actuated part !
        post_v_ref = np.matrix([0,0,0,0]).T
        post_a_ref = np.matrix([0,0,0,0]).T

        post_p_err = q[4:] - post_p_ref
        post_v_err = v[3:] - post_v_ref
        
        post_a_des = -Kp_post*post_p_err - Kd_post*post_v_err 

        z4 = np.matrix(np.zeros([4,4+4]))
        A_post  = w_post*np.hstack([Jpost,z4])
        b_post  = w_post*post_a_des # -post_drift
        #~ print "posture error \t{0}".format(np.linalg.norm(post_p_err))

        #stack all tasks
        #~ A=np.vstack([A_com,A_post])
        #~ b=np.vstack([b_com,b_post])
        A=np.vstack([A_admcom,A_post])
        b=np.vstack([b_admcom,b_post])
        #~ A=A_post
        #~ b=b_post
        #~ A=A_com
        #~ b=b_com

        #stack equality and inequality constrains
        #~ Ac = np.vstack([Aec,Aic])
        #~ bc = np.vstack([bec,bic])
        Ac=Aec
        bc=bec
        
        #formulate the least square as a quadratic problem *************
        H=(A.T*A).T + 1e-10*np.eye(A.shape[1])
        g=(A.T*b).T

        g  = np.squeeze(np.asarray(g))
        bc = np.squeeze(np.asarray(bc))
        bec = np.squeeze(np.asarray(bec))
        
        #solve it ******************************************************
        y = solve_qp(H,g,Ac.T,bc,bec.shape[0])[0]
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
        
        self.data.lkf    = f[:2].A1
        self.data.rkf    = f[2:].A1
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
