import pinocchio as se3
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
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
    def __init__(self,robot,Ky,Kz,Kp_post,Kp_com):
        self.robot = robot
        self.NQ = robot.model.nq
        self.NV = robot.model.nv
        self.NB = robot.model.nbodies
        self.RF = robot.model.getFrameId('rankle')
        self.LF = robot.model.getFrameId('lankle')
        self.RK = robot.model.frames[self.RF].parent
        self.LK = robot.model.frames[self.LF].parent
        
        self.Kp_post = Kp_post
        self.Kd_post = 2*sqrt(Kp_post)
        self.Kp_com = Kp_com
        self.Kd_com = 2*sqrt(Kp_com)
        self.Ka_com = 0 #Todo
        self.Kj_com = 0 #Todo
        self.Ky = Ky
        self.Kz = Kz
        self.data = Empty()
        com_p_ref = np.matrix([0.1,0.53]).T
        com_v_ref = np.matrix([0,0]).T
        com_a_ref = np.matrix([0,0]).T
        self.callback_com = lambda t: (com_p_ref,com_v_ref,com_a_ref)
    def solve(self,q,v,fc,dfc,t=0.0):
        robot=self.robot
        NQ,NV,NB,RF,LF,RK,LK = self.NQ,self.NV,self.NB,self.RF,self.LF,self.RK,self.LK
        
        Kp_post, Kd_post = self.Kp_post, self.Kd_post
        Kp_com,  Kd_com, Ka_com, Kj_com  = self.Kp_com, self.Kd_com, self.Ka_com, self.Kj_com
        callback_com = self.callback_com
        #~ print("\033c") # clear terminal
        # compute and extract quantities *******************************
        se3.computeAllTerms(robot.model,robot.data,q,v)
        se3.framesKinematics(robot.model,robot.data,q)
        #~ se3.rnea(robot.model,robot.data,q,v,0*v)
        m=0; 
        for mi in robot.data.mass[1:]: m+=mi 
        M  = robot.data.M        #(7,7)
        h  = robot.data.nle      #(7,1)
        Jl = se3.frameJacobian(robot.model,robot.data,LF)       
        Jr = se3.frameJacobian(robot.model,robot.data,RF)
        Jlk = se3.jacobian(robot.model,robot.data,q,LK,True,True)
        Jrk = se3.jacobian(robot.model,robot.data,q,RK,True,True)
        # Formulate dynamic constrains  (ne sert a rien...)************* todo remove and make the problem smaller
        #      |M   -Jc.T  -S.T| * [dv,f,tau].T =  -h
        Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
        S  = np.hstack([np.zeros([4,3]),np.eye(4)]) # (4,7)
        Aec = np.hstack([M ,-Jc.T,-S.T])
        bec = h 
        
        #CoM task ******************************************************
        Jcom = robot.data.Jcom[1:] #Only Y and Z
        se3.centerOfMass(robot.model,robot.data,q,v,zero(NV))
        
        # sensors
        X = np.hstack([np.eye(2),np.eye(2)]) #Constant should me moved
        X_pinv = np.linalg.pinv(X) #Constant should me moved   
        com   = robot.data.com[0][1:]
        com_v = robot.data.vcom[0][1:]
        com_a = (1/m)*X*fc
        com_j = (1/m)*X*dfc

        # ref
        com_p_ref, com_v_ref, com_a_ref = callback_com(t)
        com_j_ref = 0 # todo
        com_4d_ref = 0 # todo

        com_drift = robot.data.acom[0][1:]
        com_p_err = com   - com_p_ref
        com_v_err = com_v - com_v_ref
        com_a_err = com_a - com_a_ref
        com_j_err = com_j - com_j_ref

        com_4d_des = -Kp_com*com_p_err -Kd_com*com_v_err -Ka_com*com_a_err -Kj_com*com_j_err  +com_4d_ref
        Kl = np.eye(2)*1000 #todo 
        Kr = np.eye(2)*1000 #todo
        
        Kinv = np.hstack([np.linalg.inv(Kl),np.linalg.inv(Kr)])
        ddf_des = m*X_pinv*com_4d_des
        com_a_des = Kinv*ddf_des

        z3 = np.matrix(np.zeros([2,4+4]))

        A_com  = np.hstack([Jcom,z3])
        b_com  = com_a_des - com_drift
        #posture task  *************************************************

        Jpost = np.hstack([np.zeros([4,3]),np.eye(4)])
        post_p_ref = robot.q0[4:] #only the actuated part !
        post_v_ref = np.matrix([0,0,0,0]).T
        post_a_ref = np.matrix([0,0,0,0]).T

        post_p_err = q[4:] - post_p_ref
        post_v_err = v[3:] - post_v_ref

        post_a_des = -Kp_post*post_p_err - Kd_post*post_v_err

        z4 = np.matrix(np.zeros([4,4+4]))
        A_post  = np.hstack([Jpost,z4])
        b_post  = post_a_des # -post_drift
        #~ print "posture error \t{0}".format(np.linalg.norm(post_p_err))

        #stack all tasks
        A=np.vstack([A_com,A_post])
        b=np.vstack([b_com,b_post])
        
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
        #~ y = solve_qp(H,g)[0] # without constrains
        #~ y = np.squeeze(np.asarray(np.linalg.pinv(A)*b)) # without constrains with pinv


        dv = np.matrix(y[:7]   ).T
        f   = np.matrix(y[7:7+4]).T
        tau = np.matrix(y[7+4:] ).T
        #populate results
        self.data.com    = com.A1
        self.data.comerr = com_p_err.A1
        self.data.comref = com_p_ref.A1
        self.data.lkf    = f[:2].A1
        self.data.rkf    = f[2:].A1
        self.data.tau    = tau
        self.data.dv     = dv
        self.data.f      = f

        return np.vstack([zero(3),tau])
        

if __name__ == "__main__":
    '''benchmark TSID'''
    from hrp2_reduced import Hrp2Reduced
    from IPython import embed
    import time
    from path import pkg, urdf 
    robot = Hrp2Reduced(urdf,[pkg],loadModel=True,useViewer=False)
    tsid=TsidFlexibleContact(robot,Kp_post=10,Kp_com=30)
    niter= 3000
    t0 = time.time()
    for i in range(niter):
        tsid.solve(robot.q0,np.zeros([7,1]),np.matrix([0.,0.,0.,0.]).T,np.matrix([0.,0.,0.,0.]).T)
    print 'TSID average time = %.5f ms' % ((time.time()-t0)/(1e-3*niter)) #0.53ms
    embed()
