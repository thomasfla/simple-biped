import pinocchio as se3
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from math import pi,sqrt,cos,sin
from hrp2_reduced import Hrp2Reduced
import time 
from utils.utils_thomas import restert_viewer_server

try:
    from IPython import embed
except ImportError:
    pass

def ForceVect(force_list):
    '''Convert a list of forces into a StdVect_Force'''
    res = se3.StdVect_Force()
    res.extend(f for f in force_list)
    return res

def ForceDict(force_dict,N):
    '''Convert a dict of forces into a StdVect_Force'''
    res = se3.StdVect_Force()
    res.extend(se3.Force.Zero() if i not in force_dict else force_dict[i] for i in range(N) )
    return res

class Simu:
    '''
    Based on a Hrp2Reduced robot model, implement the simulation with spring
    contact at the 2 ankles.
    '''

    def __init__(self,robot,q0=None,dt=1e-3,ndt=10):
        '''
        Initialize from a Hrp2Reduced robot model, an initial configuration,
        a timestep dt and a number of Eurler integration step ndt.
        The <simu> method (later defined) processes <ndt> steps, each of them lasting <dt>/<ndt> seconds,
        (i.e. total integration time when calling simu is <dt> seconds).
        <q0> should be an attribute of robot if it is not given.
        '''
        self.first_iter = True
        self.tauc = np.array([0.0,0.0,0.0,0.0]) # coulomb stiction 
        self.frf = zero(6)
        self.flf = zero(6)
        self.dt  = dt       # Time step
        self.ndt = ndt      # Discretization (number of calls of step per time step)
        self.robot = robot
        self.useViewer = robot.useViewer
        
        self.NQ = robot.model.nq
        self.NV = robot.model.nv
        self.NB = robot.model.nbodies
        self.RF = robot.model.getFrameId('rankle')
        self.LF = robot.model.getFrameId('lankle')
        self.RK = robot.model.frames[self.RF].parent
        self.LK = robot.model.frames[self.LF].parent
        NQ,NV,NB,RF,LF,RK,LK = self.NQ,self.NV,self.NB,self.RF,self.LF,self.RK,self.LK

        if q0 is None: q0 = robot.q0
        se3.forwardKinematics(robot.model,robot.data,q0)
        se3.framesKinematics(robot.model,robot.data)

        self.Mrf0 = robot.data.oMf[RF].copy()  # Initial (i.e. 0-load) position of the R spring.
        self.Mlf0 = robot.data.oMf[LF].copy()  # Initial (i.e. 0-load) position of the L spring.
        
        #                      Tx    Ty     Tz      Rx   Ry    RZ
        #Hrp2 6d stiffness : (4034, 23770, 239018, 707, 502, 936)
        
        #                         Ty     Tz    Rx
        self.Krf = -np.diagflat([23770,239018.,0.])   # Stiffness of the Right spring
        self.Klf = -np.diagflat([23770,239018.,0.])   # Stiffness of the Left  spring
        self.Brf = -np.diagflat([50.,500.,0.])   # damping of the Right spring
        self.Blf = -np.diagflat([50.,500.,0.])   # damping of the Left  spring
        
        
        if self.useViewer:
            robot.viewer.addXYZaxis('world/mrf',[1.,.6,.6,1.],.03,.1)
            robot.viewer.addXYZaxis('world/mlf',[.6,.6,1.,1.],.03,.1)
            robot.viewer.applyConfiguration('world/mrf',se3ToXYZQUAT(self.Mrf0))
            robot.viewer.applyConfiguration('world/mlf',se3ToXYZQUAT(self.Mlf0))

    def step(self,q,v,tauq,dt = None):
        if dt is None: dt = self.dt
        tauq = tauq.copy()
        robot = self.robot
        NQ,NV,NB,RF,LF,RK,LK = self.NQ,self.NV,self.NB,self.RF,self.LF,self.RK,self.LK
        
        if self.first_iter: 
            self.compute_f_df_from_q_v(q,v)
            self.first_iter = False
            
        for i in range(3,7):
            if v[i]<0:
                tauq[i]+=self.tauc[i-4]
            elif v[i]>0:
                tauq[i]-=self.tauc[i-4]
                
        #~ dv  = se3.aba(robot.model,robot.data,q,v,tauq,ForceDict(self.forces,NB))
        #~ #simulazione mano! (Forces are directly in the world frame, and aba wants them in the end effector frame)
        se3.forwardKinematics(robot.model,robot.data,q,v)
        se3.computeAllTerms(robot.model,robot.data,q,v)
        se3.framesKinematics(robot.model,robot.data)
        se3.computeJacobians(robot.model,robot.data,q)
        M  = robot.data.M        #(7,7)
        h  = robot.data.nle      #(7,1)
        Jl,Jr = robot.get_Jl_Jr_world(q)
        Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
        dv = np.linalg.inv(M)*(tauq-h+Jc.T*self.f) #use last forces
        v_mean = v + 0.5*dt*dv
        v += dv*dt
        q   = se3.integrate(robot.model,q,v_mean*dt)
        self.compute_f_df_from_q_v(q,v, False)
        self.dv = dv
        return q,v
        
    def reset(self):
        self.first_iter = True
        
    def compute_f_df_from_q_v(self,q,v, compute_data = True):
        '''Compute the contact forces and them derivative via q,v and the elastic model'''
        robot = self.robot
        NQ,NV,NB,RF,LF,RK,LK = self.NQ,self.NV,self.NB,self.RF,self.LF,self.RK,self.LK
        # --- Simu
        if compute_data :
            se3.forwardKinematics(robot.model,robot.data,q,v)
            se3.computeAllTerms(robot.model,robot.data,q,v)
            se3.framesKinematics(robot.model,robot.data)
            se3.computeJacobians(robot.model,robot.data,q)

        Mrf = self.Mrf0.inverse()*robot.data.oMf[RF]
        Mlf = self.Mlf0.inverse()*robot.data.oMf[LF]
        vlf,vrf = robot.get_vlf_vrf_world(q,v)
        #extract only the free components (2d robot)
        vlf = np.vstack([vlf.linear[1:],vlf.angular[0]])
        vrf = np.vstack([vrf.linear[1:],vrf.angular[0]])
        qrf = np.vstack([Mrf.translation[1:],se3.rpy.matrixToRpy(Mrf.rotation)[0]])
        qlf = np.vstack([Mlf.translation[1:],se3.rpy.matrixToRpy(Mlf.rotation)[0]])

        frf = self.frf                                                  # Buffer where to store right force
        frf[[1,2,3]] = self.Krf*qrf + self.Brf*vrf                      # Right force in effector frame
        #~ rf0_frf = se3.Force(frf)                                        # Force in rf0 frame
        #~ rk_frf  = (robot.data.oMi[RK].inverse()*self.Mrf0).act(rf0_frf) # Spring force in R-knee frame.
        flf = self.flf                                                  # Buffer where to store left force
        flf[[1,2,3]] = self.Klf*qlf + self.Blf*vlf                      # Left force in effector frame
        #~ lf0_flf = se3.Force(flf)                                        # Force in lf0 frame
        #~ lk_flf  = (robot.data.oMi[LK].inverse()*self.Mlf0).act(lf0_flf) # Spring force in L-knee frame.

        #~ self.forces = {RK: rk_frf, LK: lk_flf}
        self.vlf = vlf
        self.vrf = vrf
        #~ 
        #~ lkMlf = robot.data.oMi[LK].inverse()*robot.data.oMf[LF]
        #~ rkMrf = robot.data.oMi[RK].inverse()*robot.data.oMf[RF]
        #~ f=np.vstack([lkMlf.actInv(self.forces[LK]).linear[1:],
                     #~ rkMrf.actInv(self.forces[RK]).linear[1:]])
        self.f=np.vstack([self.flf[1:3],self.frf[1:3]]) #  forces in the world frame
        self.df=np.vstack([(self.Klf*self.vlf)[0:2],(self.Krf*self.vrf)[0:2]])
        return self.f,self.df
        
    def simu(self,q,v,tau):
        '''Simu performs self.ndt steps each lasting self.dt/self.ndt seconds.'''
        vlf_0, vrf_0 = self.vlf.copy(), self.vrf.copy()
        # compute df based on current contact point velocities
        df_0 = np.vstack([(self.Klf*self.vlf)[0:2],(self.Krf*self.vrf)[0:2]])
        com_p_0, com_v_0, com_a_0, com_j_0 = self.robot.get_com_and_derivatives(q, v, self.f, df_0)
        f_0 = self.f.copy()
        
        for i in range(self.ndt):
            q,v = self.step(q,v,tau,self.dt/self.ndt)
            
        # compute average acc values during simulation time step
        self.acc_lf = (self.vlf - vlf_0)/self.dt
        self.acc_rf = (self.vrf - vrf_0)/self.dt
        self.df     = (self.f - f_0)/self.dt    # this df is the average over the time step
        df_end = np.vstack([(self.Klf*self.vlf)[0:2],(self.Krf*self.vrf)[0:2]])
        self.com_p, self.com_v, com_a, com_j = self.robot.get_com_and_derivatives(q, v, self.f, df_end)
        self.com_a = (self.com_v - com_v_0)/self.dt
        self.com_j = (com_a - com_a_0)/self.dt
        self.com_s = (com_j - com_j_0)/self.dt
        self.am, self.dam, self.ddam = self.robot.get_angular_momentum_and_derivatives(q, v, self.f, self.df, self.Krf[0,0], self.Krf[1,1])
        
        return q,v,self.f,self.df

    __call__ = simu


if __name__ == "__main__":
    '''Simulation using a simple PD controller.'''
    useViewer = True
    if useViewer:
        restert_viewer_server()
    from path import pkg, urdf 
    robot = Hrp2Reduced(urdf,[pkg],loadModel=True,useViewer=useViewer)
    robot.display(robot.q0)
    if useViewer:
        robot.viewer.setCameraTransform(0,[1.9154722690582275,
                                           -0.2266872227191925,
                                           0.1087859719991684,
                                           0.5243823528289795,
                                           0.518651008605957,
                                           0.4620114266872406,
                                           0.4925136864185333])

    class ControlPD:
        def __init__(self,K):
            self.setGains(K)
        def setGains(self,P,V=None):
            if V is None: V=2*sqrt(P)
            self.P,self.V = P,V
        def control(self,q,v):
            return np.vstack([zero(3),-self.P*(q-robot.q0)[4:] - self.V*v[3:]])
        __call__ = control

    simu = Simu(robot,dt=1e-3,ndt=1)

    simu.tauc = 0*np.array([1.0,1.0,1.0,1.0])
    #~ robot.model.gravity.linear=zero(3)

    def loop(q,v,niter,ndt=None,dt=None,tsleep=.9,fdisplay=10):
        t0 = time.time()
        if dt  is not None: simu.dt  = dt
        if ndt is not None: simu.ndt = ndt
        robot.display(q)
        for i in range(niter):
            q,v,f,df = simu(q,v,control(q,v))
            if not i % fdisplay:
                robot.display(q)
                #~ while((time.time()-t0)<(i*simu.dt)):
                    #~ time.sleep(0.01*simu.dt) # 1% jitter
        print 'Elapsed time = %.2f (simu time=%.2f)' % (time.time()-t0,simu.dt*niter)
        return q,v

    q = robot.q0.copy()
    v = zero(simu.NV)

    q[0]-=.01
    q[1]-=.1
    v[1]-=.001
    v[0]-=.001
    q0 = q.copy()
    v0 = v.copy()
    
    
    control = ControlPD(20000.)
    q,v = loop(q,v,int(10/simu.dt))
    embed()
