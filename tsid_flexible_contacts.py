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
        com_p_ref = np.matrix([0.,0.53]).T
        com_v_ref = np.matrix([0.,0.]).T
        com_a_ref = np.matrix([0.,0.]).T
        com_j_ref = np.matrix([0.,0.]).T
        com_s_ref = np.matrix([0.,0.]).T
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

        Jl,Jr = robot.get_Jl_Jr_world(q, False)

        # Formulate dynamic constrains  (ne sert a rien...)************* todo remove and make the problem smaller
        #      |M   -Jc.T  -S.T| * [dv,f,tau].T =  -h
        Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
        S  = np.hstack([np.zeros([4,3]),np.eye(4)]) # (4,7)
        Aec = np.vstack([np.hstack([M ,-Jc.T,-S.T]),
                         np.hstack([np.zeros([4,7]) ,np.eye(4),np.zeros([4,4])])])
        bec = np.vstack([-h,
                         fc])

        #4th order CoM via feet acceleration task **********************
        '''com_s_des -> ddf_des -> feet_a_des'''        
        
        se3.centerOfMass(robot.model,robot.data,q,v,zero(NV))
        X = np.hstack([np.eye(2),np.eye(2)])
        com   = robot.data.com[0][1:]
        com_v = robot.data.vcom[0][1:]
        com_a = (1/m)*X*fc + robot.model.gravity.linear[1:]
        com_j = (1/m)*X*dfc
        
        Mlf, Mrf = robot.get_Mlf_Mrf(q, False) 
 
        # ref
        com_p_ref, com_v_ref, com_a_ref, com_j_ref, com_s_ref = callback_com(t)
        
        # errors
        com_p_err = com   - com_p_ref
        com_v_err = com_v - com_v_ref
        com_a_err = com_a - com_a_ref
        com_j_err = com_j - com_j_ref

        Kspring = -np.matrix(np.diagflat([Ky,Kz,Ky,Kz]))   # Stiffness of the feet spring
        Kinv = np.linalg.inv(Kspring)

        com_s_des = -Kp_com*com_p_err -Kd_com*com_v_err -Ka_com*com_a_err -Kj_com*com_j_err +com_s_ref
        #~ com_s_des = -100 * np.matrix([1,1]).T #FOR TEST 
        #~ com_s_des = com_s_ref #FOR TEST only the FeedForward
        X_pinv = np.linalg.pinv(X)
        ddf_des = m*X_pinv*com_s_des
        feet_a_des = Kinv*ddf_des # This quantity is computed for debuging, but the task is now formulated in the centroidal space

        Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)

        driftLF,driftRF = robot.get_driftLF_driftRF_world(q,v, False)
        dJcdq = np.vstack([driftLF.vector[1:3],driftRF.vector[1:3]])

        #Com and Angular momentum via feet accelerations ***************
        pyl, pzl = Mlf.translation[1:].A1
        pyr, pzr = Mrf.translation[1:].A1
        fyl, fzl, fyr, fzr = fc.A1
        cy, cz     = com.A1
        dcy, dcz   = com_v.A1
        ddcy, ddcz = com_a.A1  
        dfyl, dfzl, dfyr, dfzr = dfc.A1   

        dpyl = -dfyl/Ky
        dpzl = -dfzl/Kz
        dpyr = -dfyr/Ky
        dpzr = -dfzr/Kz

        Xl = np.matrix([ 1./-Ky*fzl - (pzl - cz),
                        -1./-Kz*fyl + (pyl - cy),
                         1./-Ky*fzr - (pzr - cz),
                        -1./-Kz*fyr + (pyr - cy)])
        Nl  = fyl*ddcz - fzl*ddcy + 2*(dpyl-dcy)*dfzl - 2*(dpzl-dcz)*dfyl #checked
        Nl += fyr*ddcz - fzr*ddcy + 2*(dpyr-dcy)*dfzr - 2*(dpzr-dcz)*dfyr
        
        Xc = (1./m) * np.hstack([np.eye(2),np.eye(2)])
        Nc = np.matrix([0.,0.]).T
        
        # iam is the integral of the angular momentum approximated by the base orientation.
        # am dam ddam and dddam are the angular momentum derivatives 
                
        K_iam  = Kp_com # take the same gains as the CoM
        K_am   = Kd_com
        K_dam  = Ka_com
        K_ddam = Kj_com
        
        iam_ref, am_ref, dam_ref, ddam_ref, dddam_ref = 0.,0.,0.,0.,0.
        
        Jam = robot.get_angularMomentumJacobian(q,v)
        robotInertia = Jam[0,2] 
        
        #measurments
        iam = robotInertia * np.arcsin(q[3])
        am = (Jam*v).A1[0] # Jam*v 3d ???
        dam = (pyl-cy)*fzl                  -  (pzl-cz)*fyl                  + (pyr-cy)*fzr                 -  (pzr-cz)*fyr
        ddam = (dpyl-dcy)*fzl+(pyl-cy)*dfzl - ((dpzl-dcz)*fyl+(pzl-cz)*dfyl) + (dpyr-dcy)*fzr+(pyr-cy)*dfzr - ((dpzr-dcz)*fyr+(pzr-cz)*dfyr)

        iam_err  =  iam -  iam_ref
        am_err   =   am -   am_ref
        dam_err  =  dam -  dam_ref
        ddam_err = ddam - ddam_ref
        
        dddam_des = -K_iam*iam_err -K_am*am_err -K_dam*dam_err -K_ddam*ddam_err +dddam_ref
        
        u_des = np.vstack([com_s_des,dddam_des]) # CoM + Angular Momentum
        X = np.vstack([Xc, Xl])                  # CoM + Angular Momentum
        N = np.vstack([Nc, Nl])                  # CoM + Angular Momentum
        #~ u_des = com_s_des #CoM only
        #~ X = Xc            #CoM only
        #~ N = Nc            #CoM only        
        
        A_feet = X*Kspring*Jc
        b_feet = u_des - N - X*Kspring*dJcdq
        
        A_feet = np.hstack([A_feet ,np.zeros([X.shape[0],4+4])]) # feed zero for the variable not in the problem (forces and torques)

        #posture task  *************************************************

        #~ Jpost = np.hstack([np.zeros([4,3]),np.eye(4)])
        Jpost = np.eye(7)
        post_p_ref = robot.q0 #[4:] #only the actuated part !
        post_v_ref = np.matrix([0.,0.,0.,0.,0.,0.,0.]).T
        post_a_ref = np.matrix([0.,0.,0.,0.,0.,0.,0.]).T
        post_p_err = np.matrix([0.,0.,0.,0.,0.,0.,0.]).T
        
        post_p_err[:2] = q[:2] - post_p_ref[:2] # Y-Z error
        post_p_err[2]  = np.arcsin(q[3]) - np.arcsin(post_p_ref[3]) # Angular error
        post_p_err[3:] = q[4:] - post_p_ref[4:] # Actuation error
        post_v_err = v - post_v_ref

        post_a_des = -Kp_post*post_p_err - Kd_post*post_v_err

        z4 = np.matrix(np.zeros([7,4+4]))
        A_post  = w_post*np.hstack([Jpost,z4]) 
        b_post  = w_post*post_a_des   
        #~ A_post[2,2] = 0. # For tests, remove angular regulation 
        #~ b_post[2] = 0.        
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
        H=(A.T*A).T + 1e-8*np.eye(A.shape[1])
        g=(A.T*b).T
        
        g  = np.squeeze(np.asarray(g))
        bc = np.squeeze(np.asarray(bc))
        bec = np.squeeze(np.asarray(bec))
        #solve it ******************************************************
        y_QP = solve_qp(H,g,Ac.T,bc,bec.shape[0])[0]
        
        # Solve with inverse *******************************************
        if 0:        
            Mu = robot.data.M[:3]
            Ma = robot.data.M[3:]
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
        self.data.robotInertia = robotInertia
        
        self.data.iam  = iam
        self.data.am   = am
        self.data.dam  = dam
        self.data.ddam = ddam
        self.data.dddam_des = dddam_des
        
        self.data.lkf    = f[:2].A1
        self.data.rkf    = f[2:].A1
        self.data.tau    = tau
        self.data.dv     = dv
        self.data.f      = f

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
    np.set_printoptions(precision=3)
    np.set_printoptions(linewidth=200)
    robot = Hrp2Reduced(urdf,[pkg],loadModel=True,useViewer=False)
    tsid=TsidFlexibleContact(robot,Ky=23770.,Kz=239018.,w_post=0e-3,Kp_post=1.,Kp_com=1., Kd_com=1., Ka_com=1., Kj_com=1.)
    niter= 3000
    t0 = time.time()
    for i in range(niter):
        tsid.solve(robot.q0,np.zeros([7,1]),np.matrix([0.,0.,0.,0.]).T,np.matrix([0.,0.,0.,0.]).T)
    print 'TSID average time = %.5f ms' % ((time.time()-t0)/(1e-3*niter)) #0.53ms
    embed()
