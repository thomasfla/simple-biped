import pinocchio as se3
from pinocchio.utils import *
from math import pi,sqrt,cos,sin
from hrp2_reduced import Hrp2Reduced
import time 
from simu import Simu

try:
    from IPython import embed
except:
    pass

#pkg = '/opt/openrobots/share/hrp2-14/'
#urdf = '/home/tflayols/devel/openrobots/src/sot-torque-control/hrp2-torque-control/share/metapod/data/hrp2_14/hrp2_14.urdf'
#~ pkg = '/home/nmansard/src/sot_versions/groovy/ros/stacks/hrp2/'
#~ urdf = '/home/nmansard/src/pinocchio/pinocchio/models/hrp2014.urdf'
from path import pkg, urdf 

robot = Hrp2Reduced(urdf,[pkg],loadModel=True)
robot.display(robot.q0)
robot.viewer.setCameraTransform(0,[1.9154722690582275,
                                   -0.2266872227191925,
                                   0.1087859719991684,
                                   0.5243823528289795,
                                   0.518651008605957,
                                   0.4620114266872406,
                                   0.4925136864185333])

simu = Simu(robot,dt=1e-3,ndt=5)
NQ,NV,NB,RF,LF,RK,LK = simu.NQ,simu.NV,simu.NB,simu.RF,simu.LF,simu.RK,simu.LK

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


class Stab:
    def __init__(self):
        pass

q = robot.q0.copy()
v = zero(NV)
v = rand(NV)
#se3.crba(robot.model,robot.data,q)
se3.computeAllTerms(robot.model,robot.data,q,v)

M  = robot.data.M
b  = robot.data.nle

Jr = se3.frameJacobian(robot.model,robot.data,q,RF,True,False)
Jl = se3.frameJacobian(robot.model,robot.data,q,LF,True,False)

ark = robot.data.a[RK]
ar  = robot.model.frames[RF].placement.inverse()*ark
alk = robot.data.a[LK]
al  = robot.model.frames[LF].placement.inverse()*ark

Jcom = robot.data.Jcom
se3.centerOfMass(robot.model,robot.data,q,v,zero(NV))
acom = robot.data.acom[0]

try:
    embed()
except:
    pass
