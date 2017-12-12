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
    res = se3.StdVect_Force()
    res.extend(f for f in force_list)
    return res

def ForceDict(force_dict,N):
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

se3.forwardKinematics(robot.model,robot.data,robot.q0)
se3.framesKinematics(robot.model,robot.data)
Mrf0 = robot.data.oMf[RF].copy()
Mlf0 = robot.data.oMf[LF].copy()
#Krf = -np.diagflat([5000.,5000.,5000.])
#Klf = -np.diagflat([5000.,5000.,5000.])
Krf = -np.diagflat([200000.,200000.,600.])
Klf = -np.diagflat([200000.,200000.,600.])

robot.viewer.addXYZaxis('world/mrf',[1.]*4,.03,.1)
robot.viewer.addXYZaxis('world/mlf',[1.]*4,.03,.1)
robot.viewer.applyConfiguration('world/mrf',se3ToXYZQUAT(Mrf0))
robot.viewer.applyConfiguration('world/mlf',se3ToXYZQUAT(Mlf0))

# robot.viewer.addSphere('world/redball0', 0.05,[1.,.5,.5,1.])
# robot.viewer.addBox('world/redsole0', 0.15,.06,.01,[1.,.5,.5,1.])
# robot.viewer.applyConfiguration('world/redsole0',se3ToXYZQUAT(Mrf0))
# robot.viewer.applyConfiguration('world/redball0',se3ToXYZQUAT(Mrf0))

#visuals.append( Visual('world/red'+name+'sole',jointId,placement) )

q = robot.q0.copy()
v = zero(NV)

q[0]-=.02
q[1]-=.02

class Simu:
    def __init__(self,dt=1e-3):
        self.frf = zero(6)
        self.flf = zero(6)
        self.dt  = dt

    def simu(self,q,v,tau):
        self.vq = v.copy()
        self.q  = q.copy()

        # --- Simu
        se3.forwardKinematics(robot.model,robot.data,q,v)
        se3.framesKinematics(robot.model,robot.data)
        
        Mrf = Mrf0.inverse()*robot.data.oMf[RF]
        vrf = robot.model.frames[RF].placement.inverse()*robot.data.v[RK]
        Mlf = Mlf0.inverse()*robot.data.oMf[LF]
        vlf = robot.model.frames[LF].placement.inverse()*robot.data.v[LK]

        qrf = np.vstack([Mrf.translation[1:],se3.rpy.matrixToRpy(Mrf.rotation)[0]])
        qlf = np.vstack([Mlf.translation[1:],se3.rpy.matrixToRpy(Mlf.rotation)[0]])

        frf = self.frf
        frf[[1,2,3]] = Krf*qrf
        rf0_frf = se3.Force(frf) # Force in rf0 frame
        rk_frf  = (robot.data.oMi[RK].inverse()*Mrf0).act(rf0_frf)  # Spring force in rk frame.
        flf = self.flf
        flf[[1,2,3]] = Klf*qlf
        lf0_flf = se3.Force(flf) # Force in lf0 frame
        lk_flf  = (robot.data.oMi[LK].inverse()*Mlf0).act(lf0_flf)  # Spring force in lk frame.

        self.forces = ForceDict({ RK: rk_frf, LK: lk_flf},NB)

        self.aq  = se3.aba(robot.model,robot.data,q,v,tau,self.forces)
        self.vq += self.aq*self.dt
        self.q   = se3.integrate(robot.model,q,self.vq*self.dt)
        return self.q,self.vq

    __call__ = simu

class ControlPD:
    def __init__(self,K):
        self.setGains(K)
    def setGains(self,P,V=None):
        if V is None: V=2*sqrt(P)
        self.P,self.V = P,V
    def control(self,q,v):
        return np.vstack([zero(3),-self.P*(q-robot.q0)[4:] - self.V*v[3:]])
    __call__ = control

simu = Simu()
control = ControlPD(100000)
#robot.model.gravity.linear=zero(3)

def loop(q,v,niter,ndt=50,dt=5e-3,tsleep=.9):
    t0 = time.time()
    simu.dt = dt/ndt
    for i in range(niter):
        robot.display(q)
        time.sleep(tsleep*dt)
        for j in range(ndt):
            q,v = simu(q,v,control(q,v))
    print 'Elapsed time = %.2f (simu time=%.2f)' % (time.time()-t0,dt*niter)
    return q,v


#boolean addArrow(in string arrowName, in float radius, in float length, in Color RGBAcolorid) raises (Error);
#boolean resizeArrow(in string capsuleName,in float radius, in float length) raises (Error);
