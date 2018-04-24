import pinocchio as se3
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
#~ from pinocchio.utils import eye, zero, isapprox
from math import pi,sqrt,cos,sin
from quadprog import solve_qp
from IPython import embed
from utils_thomas import restert_viewer_server
from logger import Logger
from filters import FIR1LowPass
import matplotlib.pyplot as plt

class Empty:
    pass
    
class TsidFlexibleContact:
    def __init__(self,robot,Ky,Kz,w_post,Kp_post,Kp_com, Kd_com, Ka_com, Kj_com):
        self.robot = robot
        self.NQ = robot.model.nq
        self.NV = robot.model.nv
        self.NB = robot.model.nbodies
        self.RF = robot.model.getFrameId('rankle')
        self.LF = robot.model.getFrameId('lankle')
        self.RK = robot.model.frames[self.RF].parent
        self.LK = robot.model.frames[self.LF].parent
        
        self.w_post = w_post
        self.Kp_post = Kp_post
        self.Kd_post = 2*sqrt(Kp_post)
        self.Kp_com = Kp_com
        self.Kd_com = Kd_com
        self.Ka_com = Ka_com
        self.Kj_com = Kj_com
        self.Ky = Ky
        self.Kz = Kz
        self.data = Empty()
        com_p_ref = np.matrix([0,0.53]).T
        com_v_ref = np.matrix([0,0]).T
        com_a_ref = np.matrix([0,0]).T
        com_j_ref = np.matrix([0,0]).T
        com_s_ref = np.matrix([0,0]).T
        self.callback_com = lambda t: (com_p_ref,com_v_ref,com_a_ref,com_j_ref,com_s_ref )
    def solve(self,q,v,fc,dfc,t=0.0):
        robot=self.robot
        NQ,NV,NB,RF,LF,RK,LK = self.NQ,self.NV,self.NB,self.RF,self.LF,self.RK,self.LK
        w_post = self.w_post
        Kp_post, Kd_post = self.Kp_post, self.Kd_post
        Kp_com,  Kd_com, Ka_com, Kj_com  = self.Kp_com, self.Kd_com, self.Ka_com, self.Kj_com
        Ky = self.Ky 
        Kz = self.Kz 
        callback_com = self.callback_com

        se3.computeAllTerms(robot.model,robot.data,q,v)
        se3.framesKinematics(robot.model,robot.data,q)
        se3.rnea(robot.model,robot.data,q,v,0*v)
        m = robot.data.mass[0]
        M  = robot.data.M        #(7,7)
        h  = robot.data.nle      #(7,1)

        Jl,Jr = robot.get_Jl_Jr_world(q)

        # Formulate dynamic constrains  (ne sert a rien...)************* todo remove and make the problem smaller
        #      |M   -Jc.T  -S.T| * [dv,f,tau].T =  -h
        Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
        S  = np.hstack([np.zeros([4,3]),np.eye(4)]) # (4,7)
        Aec = np.vstack([np.hstack([M ,-Jc.T,-S.T]),
                         np.hstack([np.zeros([4,7]) ,np.eye(4),np.zeros([4,4])])])
        bec = np.vstack([-h,
                         fc])
        #~ Aec =np.hstack([M ,-Jc.T,-S.T])
        #~ bec = -h
        
        #4th order CoM via feet acceleration task **********************
        '''com_s_des -> ddf_des -> feet_a_des'''        
        
        # sensors
        se3.centerOfMass(robot.model,robot.data,q,v,zero(NV))
        X = np.hstack([np.eye(2),np.eye(2)]) #Constant should be moved
        #~ X_pinv = np.linalg.pinv(X) #Constant should be moved   
        #~ X  = np.matrix([[1,        0       , 1       , 0       ],
                        #~ [0       , 1-q[3,0], 0       , 1+q[3,0]]])

        X_pinv = np.linalg.pinv(X) #Constant should be moved   
        
        com   = robot.data.com[0][1:]
        com_v = robot.data.vcom[0][1:]
        com_a = (1/m)*X*fc + robot.model.gravity.linear[1:]
        com_j = (1/m)*X*dfc

        # ref
        com_p_ref, com_v_ref, com_a_ref, com_j_ref, com_s_ref = callback_com(t)
        
        # errors
        com_p_err = com   - com_p_ref
        com_v_err = com_v - com_v_ref
        com_a_err = com_a - com_a_ref
        com_j_err = com_j - com_j_ref

        Kspring = -np.diagflat([Ky,Kz,Ky,Kz])   # Stiffness of the feet spring
        Kinv = np.linalg.inv(Kspring)
        #~ print q
        com_s_des = -Kp_com*com_p_err -Kd_com*com_v_err -Ka_com*com_a_err -Kj_com*com_j_err  +com_s_ref
        #~ com_s_des = -100 * np.matrix([1,1]).T #FOR TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #~ com_s_des = com_s_ref #FOR TEST only the FeedForward !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ddf_des = m*X_pinv*com_s_des
        feet_a_des = Kinv*ddf_des
        #~ print feet_a_des
        Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
        z1 = np.matrix(np.zeros([4,4]))
        z2 = np.matrix(np.zeros([4,4]))


        driftLF,driftRF = robot.get_driftLF_driftRF_world(q,v)
        dJcdq = np.vstack([driftLF.vector[1:3],driftRF.vector[1:3]])
        
        A_feet  = np.hstack([Jc,z1,z2]) #rename Jc into Jfeet?
        b_feet  = feet_a_des - dJcdq #rename dJcdq into feet_drift?
        
        #Quick fix, let free the forces repartition betwin the feet by adding left and right snap task
        A_feet = A_feet[2:] + A_feet[:2]
        b_feet = b_feet[2:] + b_feet[:2]
        #posture task  *************************************************

        #~ Jpost = np.hstack([np.zeros([4,3]),np.eye(4)])
        Jpost = np.eye(7)
        post_p_ref = robot.q0 #[4:] #only the actuated part !
        post_v_ref = np.matrix([0,0,0,0,0,0,0]).T
        post_a_ref = np.matrix([0,0,0,0,0,0,0]).T
        post_p_err = 0*post_v_ref #redo
        
        post_p_err[:2] = q[:2] - post_p_ref[:2] # Y-Z error
        post_p_err[2]  = np.arccos(q[2]) - np.arccos(post_p_ref[2]) # Angular error
        post_p_err[3:] = q[4:] - post_p_ref[4:] # Actuation error
        post_v_err = v - post_v_ref

        post_a_des = -Kp_post*post_p_err - Kd_post*post_v_err

        z4 = np.matrix(np.zeros([7,4+4]))
        A_post  = w_post*np.hstack([Jpost,z4]) 
        b_post  = w_post*post_a_des
        #~ print "posture error \t{0}".format(np.linalg.norm(post_p_err))

        #stack all tasks
        A=np.vstack([A_feet, A_post])
        b=np.vstack([b_feet, b_post])
        
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
        y_QP = solve_qp(H,g,Ac.T,bc,bec.shape[0])[0]

        # Solve with inverse *******************************************
        Mu = robot.data.M[:3]
        Ma = robot.data.M[3:]
        JuT = Jc.T[:3]
        hu = h[:3]
        A_ = np.vstack([ Mu,
                         Jc,
                         w_post*Jpost ])
                        
        b_ = np.vstack([ JuT*fc - hu,
                         feet_a_des - dJcdq,
                         w_post*post_a_des  ])

        #~ A_ = np.vstack([ Mu,
                         #~ Jc])
                        #~ 
        #~ b_ = np.vstack([ JuT*fc - hu,
                         #~ feet_a_des - dJcdq ])              
        
        if 0:
            dv_PINV = np.linalg.pinv(A_)*b_
            tau_PINV = S*(M*dv_PINV+h-Jc.T*fc)  #get tau_des from dv_des and measured contact forces:
            f_PINV=fc
            y_PINV = np.vstack([dv_PINV,f_PINV,tau_PINV]).A1
            assert isapprox(y_PINV,y_QP)

        dv_QP  = np.matrix(y_QP[:7]   ).T
        f_QP   = np.matrix(y_QP[7:7+4]).T
        tau_QP = np.matrix(y_QP[7+4:] ).T
        
        
        dv  = dv_QP
        f   = f_QP
        tau = tau_QP
        #~ dv  = dv_PINV
        #~ f   = f_PINV
        #~ tau = tau_PINV
        #populate results
        self.data.lf_a_des = feet_a_des[:2]
        self.data.rf_a_des = feet_a_des[2:]
        self.data.com_p  = com.A1
        self.data.com_v  = com_v.A1
        self.data.com_a  = com_a.A1
        self.data.com_j  = com_j.A1
        self.data.com_s_des = com_s_des.A1
        self.data.com_p_err = com_p_err.A1
        self.data.com_v_err = com_v_err.A1
        self.data.com_a_err = com_a_err.A1
        self.data.com_j_err = com_j_err.A1
        self.data.comref = com_p_ref.A1
        self.data.lkf    = f[:2].A1
        self.data.rkf    = f[2:].A1
        self.data.tau    = tau
        self.data.dv     = dv
        self.data.f      = f
        #~ print 'CONTROLLER'
        #~ embed()
        #~ print 'TSID Drift LF:'
        #~ print driftLF
        #~ print 'TSID dv:'
        #~ print dv
        #~ print Jl*dv
        #~ print feet_a_des 
        #~ print Jc*dv + dJcdq

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
    from IPython import embed
    import time
    from path import pkg, urdf 
    robot = Hrp2Reduced(urdf,[pkg],loadModel=True,useViewer=False)
    tsid=TsidFlexibleContact(robot,Ky=23770.,Kz=239018.,w_post=0e-3,Kp_post=1.,Kp_com=1., Kd_com=1., Ka_com=1., Kj_com=1.)
    niter= 3000
    t0 = time.time()
    for i in range(niter):
        tsid.solve(robot.q0,np.zeros([7,1]),np.matrix([0.,0.,0.,0.]).T,np.matrix([0.,0.,0.,0.]).T)
    print 'TSID average time = %.5f ms' % ((time.time()-t0)/(1e-3*niter)) #0.53ms
    embed()
