import pinocchio as se3
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from math import pi,sqrt,cos,sin
from hrp2_reduced import Hrp2Reduced
import time 

pkg = '/home/nmansard/src/sot_versions/groovy/ros/stacks/hrp2/'
urdf = '/home/nmansard/src/pinocchio/pinocchio/models/hrp2014.urdf'
robot = Hrp2Reduced(urdf,[pkg],loadModel=True)
robot.display(robot.q0)
robot.viewer.setCameraTransform(0,[1.9154722690582275,
                                   -0.2266872227191925,
                                   0.1087859719991684,
                                   0.5243823528289795,
                                   0.518651008605957,
                                   0.4620114266872406,
                                   0.4925136864185333])

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

NQ = robot.model.nq
NV = robot.model.nv
NB = robot.model.nbodies
RF = robot.model.getFrameId('rankle')
LF = robot.model.getFrameId('lankle')
RK = robot.model.frames[RF].parent
LK = robot.model.frames[LF].parent


# robot.viewer.addSphere('world/redball0', 0.05,[1.,.5,.5,1.])
# robot.viewer.addBox('world/redsole0', 0.15,.06,.01,[1.,.5,.5,1.])
# robot.viewer.applyConfiguration('world/redsole0',se3ToXYZQUAT(Mrf0))
# robot.viewer.applyConfiguration('world/redball0',se3ToXYZQUAT(Mrf0))


class Simu:
    def __init__(self,robot,q0=None,dt=1e-3,ndt=10):
        self.frf = zero(6)
        self.flf = zero(6)
        self.dt  = dt       # Time step
        self.ndt = ndt      # Discretization (number of calls of step per time step)
        self.robot = robot

        if q0 is None: q0 = robot.q0
        se3.forwardKinematics(robot.model,robot.data,q0)
        se3.framesKinematics(robot.model,robot.data)

        self.Mrf0 = robot.data.oMf[RF].copy()
        self.Mlf0 = robot.data.oMf[LF].copy()

        self.Krf = -np.diagflat([20000.,200000.,00.])
        self.Klf = -np.diagflat([20000.,200000.,00.])

        robot.viewer.addXYZaxis('world/mrf',[1.,.6,.6,1.],.03,.1)
        robot.viewer.addXYZaxis('world/mlf',[.6,.6,1.,1.],.03,.1)
        robot.viewer.applyConfiguration('world/mrf',se3ToXYZQUAT(self.Mrf0))
        robot.viewer.applyConfiguration('world/mlf',se3ToXYZQUAT(self.Mlf0))

    def step(self,q,vq,tauq,dt = None):
        if dt is None: dt = self.dt
        robot = self.robot

        # --- Simu
        se3.forwardKinematics(robot.model,robot.data,q,v)
        se3.framesKinematics(robot.model,robot.data)
        
        Mrf = self.Mrf0.inverse()*robot.data.oMf[RF]
        vrf = robot.model.frames[RF].placement.inverse()*robot.data.v[RK]
        Mlf = self.Mlf0.inverse()*robot.data.oMf[LF]
        vlf = robot.model.frames[LF].placement.inverse()*robot.data.v[LK]

        qrf = np.vstack([Mrf.translation[1:],se3.rpy.matrixToRpy(Mrf.rotation)[0]])
        qlf = np.vstack([Mlf.translation[1:],se3.rpy.matrixToRpy(Mlf.rotation)[0]])

        frf = self.frf
        frf[[1,2,3]] = self.Krf*qrf
        rf0_frf = se3.Force(frf) # Force in rf0 frame
        rk_frf  = (robot.data.oMi[RK].inverse()*self.Mrf0).act(rf0_frf)  # Spring force in rk frame.
        flf = self.flf
        flf[[1,2,3]] = self.Klf*qlf
        lf0_flf = se3.Force(flf) # Force in lf0 frame
        lk_flf  = (robot.data.oMi[LK].inverse()*self.Mlf0).act(lf0_flf)  # Spring force in lk frame.

        self.forces = ForceDict({ RK: rk_frf, LK: lk_flf},NB)

        aq  = se3.aba(robot.model,robot.data,q,vq,tauq,self.forces)
        vq += aq*dt
        q   = se3.integrate(robot.model,q,vq*dt)

        self.aq = aq
        return q,vq

    def simu(self,q,v,tau):
        for i in range(self.ndt):
            q,v = self.step(q,v,tau,self.dt/self.ndt)
        return q,v

    __call__ = simu


if __name__ == "__main__":
    class ControlPD:
        def __init__(self,K):
            self.setGains(K)
        def setGains(self,P,V=None):
            if V is None: V=2*sqrt(P)
            self.P,self.V = P,V
        def control(self,q,v):
            return np.vstack([zero(3),-self.P*(q-robot.q0)[4:] - self.V*v[3:]])
        __call__ = control

    simu = Simu(robot,dt=1e-3,ndt=5)
    control = ControlPD(10000)

    #robot.model.gravity.linear=zero(3)

    def loop(q,v,niter,ndt=None,dt=None,tsleep=.9,fdisplay=1):
        t0 = time.time()
        if dt  is not None: simu.dt  = dt
        if ndt is not None: simu.ndt = ndt
        robot.display(q)
        for i in range(niter):
            q,v = simu(q,v,control(q,v))
            if not i % fdisplay:
                robot.display(q)
                time.sleep(tsleep*simu.dt)
        print 'Elapsed time = %.2f (simu time=%.2f)' % (time.time()-t0,simu.dt*niter)
        return q,v

    def loopdg(q,v,niter,ndt=50,dt=5e-3,tsleep=.9,fdisplay=1):
        t0 = time.time()
        simu.dt  = dt/ndt
        simu.ndt = 1
        for i in range(niter):
            if not i % fdisplay:
                robot.display(q)
                time.sleep(tsleep*dt)
            for j in range(ndt):
                q,v = simu(q,v,control(q,v))
        print 'Elapsed time = %.2f (simu time=%.2f)' % (time.time()-t0,dt*niter)
        return q,v

    q = robot.q0.copy()
    v = zero(NV)

    q[0]-=.02
    q[1]-=.02

    q0 = q.copy()
    v0 = v.copy()
