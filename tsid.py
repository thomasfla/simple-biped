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
    
class Tsid:
    def __init__(self,robot,Kp_post,Kp_com):
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
        self.data = Empty()
        com_p_ref = np.matrix([0.1,0.53]).T
        com_v_ref = np.matrix([0,0]).T
        com_a_ref = np.matrix([0,0]).T
        self.callback_com = lambda t: (com_p_ref,com_v_ref,com_a_ref)
    def setGains(self,P,V=None):
        if V is None: 
            V=2*sqrt(P)
            self.P,self.V = P,V
    def solve(self,q,v,t=0.0):
        robot=self.robot
        NQ,NV,NB,RF,LF,RK,LK = self.NQ,self.NV,self.NB,self.RF,self.LF,self.RK,self.LK
        
        Kp_post, Kd_post = self.Kp_post, self.Kd_post
        Kp_com,  Kd_com  = self.Kp_com,  self.Kd_com
        callback_com = self.callback_com
        #~ print("\033c") # clear terminal
        # compute and extract quantities *******************************
        se3.computeAllTerms(robot.model,robot.data,q,v)
        se3.framesKinematics(robot.model,robot.data,q)
        #~ se3.rnea(robot.model,robot.data,q,v,0*v)
        M  = robot.data.M        #(7,7)
        h  = robot.data.nle      #(7,1)
        Jl = se3.frameJacobian(robot.model,robot.data,LF)       
        Jr = se3.frameJacobian(robot.model,robot.data,RF)
        Jlk = se3.jacobian(robot.model,robot.data,q,LK,True,True)
        Jrk = se3.jacobian(robot.model,robot.data,q,RK,True,True)

        # Formulate contact and dynamic constrains *********************
        #        7    4     4       7  4  4
        # 6   |Jc    0      0 | * [dv,f,tau].T =  |-dJc*dq|
        # 7   |M   -Jc.T  -S.T|                   |  -h   |

        Jk = np.vstack([Jlk[1:3],Jrk[1:3]])  # (4, 7)
        Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
        S  = np.hstack([np.zeros([4,3]),np.eye(4)]) # (4,7)
        z1 = np.matrix(np.zeros([4,4]))
        z2 = np.matrix(np.zeros([4,4]))

        #~ driftLF, driftRF = robot.data.a[LF] , robot.data.a[RF]
        #~ v_LF,    v_RF    = robot.data.v[LF] , robot.data.v[RF]
        v_LF = robot.model.frames[LF].placement.actInv(robot.data.v[robot.model.frames[LF].parent]).copy()
        a_LF = robot.model.frames[LF].placement.actInv(robot.data.a[robot.model.frames[LF].parent]).copy()
        v_RF = robot.model.frames[RF].placement.actInv(robot.data.v[robot.model.frames[RF].parent]).copy()
        a_RF = robot.model.frames[RF].placement.actInv(robot.data.a[robot.model.frames[RF].parent]).copy()
        driftLF = a_LF
        driftRF = a_RF
        driftLF.linear += np.cross(v_LF.angular.T, v_LF.linear.T).T
        driftRF.linear += np.cross(v_RF.angular.T, v_RF.linear.T).T
        dJcdq = np.vstack([driftLF.vector[1:3],driftRF.vector[1:3]])

        Aec = np.vstack([ np.hstack([Jc, z1  ,z2])  ,
                          np.hstack([M ,-Jc.T,-S.T])  ])  # -Jk.T
        bec = np.vstack([-dJcdq,-h])
        #only contacts
        #~ Ac = np.hstack([Jc, z1  ,z2]) 
        #~ bc = -dJcdq 
        
        #only dynamics
        #~ Ac = np.hstack([M ,-Jc.T,-S.T])
        #~ bc = h 

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

        com_p_ref, com_v_ref, com_a_ref = callback_com(t)
        se3.centerOfMass(robot.model,robot.data,q,v,zero(NV))
        com  = robot.data.com[0][1:]
        vcom = robot.data.vcom[0][1:]
        com_drift = robot.data.acom[0][1:]
        com_p_err =  com - com_p_ref
        com_v_err = vcom - com_v_ref

        com_a_des = -Kp_com*com_p_err -Kd_com*com_v_err +com_a_ref

        z3 = np.matrix(np.zeros([2,4+4]))

        A_com  = np.hstack([Jcom,z3])
        b_com  = com_a_des - com_drift
        #~ if not log_index%100 :
            #~ print "com error \t{0}".format(np.linalg.norm(com_p_err))
        #posture task  *************************************************

        Jpost = np.hstack([np.zeros([4,3]),np.eye(4)])
        post_p_ref = robot.q0[4:] #only the actuated part !
        post_v_ref = np.matrix([0,0,0,0]).T
        post_a_ref = np.matrix([0,0,0,0]).T

        post_p_err = q[4:] - post_p_ref
        post_v_err = v[3:] - post_v_ref

        post_a_des = -Kp_post*post_p_err - Kd_post*post_v_err #No a_ref here

        z4 = np.matrix(np.zeros([4,4+4]))
        A_post  = np.hstack([Jpost,z4])
        b_post  = post_a_des # -post_drift
        #~ print "posture error \t{0}".format(np.linalg.norm(post_p_err))

        #stack all tasks
        A=np.vstack([A_com,A_post])
        b=np.vstack([b_com,b_post])
        
        #stack equality and inequality constrains
        Ac = np.vstack([Aec,Aic])
        bc = np.vstack([bec,bic])
        
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
        self.data.com    = com.A1
        self.data.com_p_err = com_p_err.A1
        self.data.com_v_err = com_v_err.A1
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
    tsid=Tsid(robot,Kp_post=10,Kp_com=30)
    niter= 3000
    t0 = time.time()
    for i in range(niter):
        tsid.solve(robot.q0,np.zeros([7,1]))
    print 'TSID average time = %.5f ms' % ((time.time()-t0)/(1e-3*niter)) #0.53ms
    embed()
