import pinocchio as se3
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from math import pi,sqrt,cos,sin
from hrp2_reduced import Hrp2Reduced
import time 
from simu import Simu, ForceDict
from quadprog import solve_qp
from IPython import embed
from utils_thomas import restert_viewer_server
from logger import Logger
from filters import FIR1LowPass, BALowPass
import matplotlib.pyplot as plt
import quadprog
from tsid import Tsid
from path import pkg, urdf 
useViewer = False
if useViewer:
    restert_viewer_server()


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
#Simulation parameters
dt  = 1e-3
ndt = 10
simulation_time = 4.0

#robot parameters
tauc = np.array([5.,5.,5.,5.])#coulomb friction
Ky = 23770.
Kz = 239018.
By = 50.
Bz = 500.
Kspring = -np.diagflat([Ky,Kz,0.])   # Stiffness of the feet spring
Bspring = -np.diagflat([By,Bz,0.])        # damping of the feet spring

#Controller parameters
fc      = 50 #np.inf #cutoff frequency of the Force filmer
Ktau    = 2.0
Kp_post = 10
Kp_com  = 30
FTSfilter = FIR1LowPass(np.exp(-2*np.pi*fc*dt)) #Force sensor filter
b, a = np.array ([0.00554272,  0.01108543,  0.00554272]), np.array([1., -1.77863178,  0.80080265])
#~ FTSfilter = BALowPass(b,a,"butter_lp_filter_Wn_05_N_2")  


log_size = int(simulation_time / dt)       #max simulation samples
log_com    = np.zeros([log_size,2])+np.nan #Centrer of mass
log_comref = np.zeros([log_size,2])+np.nan #Centrer of mass reference
log_comerr = np.zeros([log_size,2])+np.nan #Centrer of mass error
log_lkf    = np.zeros([log_size,2])+np.nan #left  knee force
log_rkf    = np.zeros([log_size,2])+np.nan #right knee force
log_lkf_sensor = np.zeros([log_size,2])+np.nan #left  knee force from FT/sensor (from deformation)
log_rkf_sensor = np.zeros([log_size,2])+np.nan #right knee force from FT/sensor (from deformation)
log_tau_des = np.zeros([log_size,4])+np.nan  #torques desired by TSID
log_tau_est = np.zeros([log_size,4])+np.nan  #torques estimated by contact forces
log_tau_ctrl = np.zeros([log_size,4])+np.nan #torque sent to the motors
log_t        = np.zeros([log_size,1])+np.nan # time
log_index = 0  


simu = Simu(robot,dt=dt,ndt=ndt)
simu.tauc = tauc
simu.Krf = Kspring
simu.Klf = Kspring
simu.Brf = Bspring
simu.Blf = Bspring
Kspring
tsid=Tsid(robot,Kp_post,Kp_com)
NQ,NV,NB,RF,LF,RK,LK = simu.NQ,simu.NV,simu.NB,simu.RF,simu.LF,simu.RK,simu.LK
def loop(q,v,f,niter,ndt=None,dt=None,tsleep=.9,fdisplay=1):
    global log_index
    t0 = time.time()
    if dt  is not None: simu.dt  = dt
    if ndt is not None: simu.ndt = ndt
    robot.display(q)
    for i in range(niter):
        log_index = i
        q,v,f = simu(q,v,control(q,v,f))
        log_lkf_sensor[log_index] =f[:2].A1
        log_rkf_sensor[log_index] =f[2:].A1
        if not i % fdisplay:
            robot.display(q)
            while((time.time()-t0)<(i*simu.dt)):
                time.sleep(0.01*simu.dt) # 1% jitter
        
    print 'Elapsed time = %.2f (simu time=%.2f)' % (time.time()-t0,simu.dt*niter)
    return q,v
    
def traj_sinusoid(t,start_position,stop_position,travel_time):
    # a cos(bt) + c
    a=-(stop_position-start_position)*0.5
    b = np.pi/travel_time
    c = start_position+(stop_position-start_position)*0.5
    
    p =     a*np.cos(b*t) + c
    v =  -b*a*np.sin(b*t)
    a =-b*b*a*np.cos(b*t)
    return p,v,a
# check derivatives
#~ eps = 1e-5
#~ p1,v1,a1 =  traj_sinusoid(0.123   ,1.1,2.2,3.3)
#~ p2,v2,a2 =  traj_sinusoid(0.123+eps,1.1,2.2,3.3)
#~ print "v: {0} dp: {1}".format(v1,(p2-p1)/eps)
#~ print "a: {0} dv: {1}".format(a1,(v2-v1)/eps)
#~ 

def controller(q,v,f):
    ''' Take the sensor data (position, velocity and contact forces) 
    and generate a torque command'''
    t=log_index*simu.dt
    #filter forces
    f_filtered = FTSfilter.update(f)
    tsid.solve(q,v,t)
    dv      = tsid.data.dv
    tau_des = tsid.data.tau
    
    log_com[log_index]    = tsid.data.com
    log_comerr[log_index] = tsid.data.comerr
    log_comref[log_index] = tsid.data.comref
    log_lkf[log_index]    = tsid.data.lkf
    log_rkf[log_index]    = tsid.data.rkf
    
    #estimate actual torque from contact forces
    tau_est  = compute_torques_from_dv_and_forces(dv,f_filtered)
    tau_ctrl = torque_controller(tau_des,tau_est)
    log_tau_ctrl[log_index] = tau_ctrl.A1
    log_tau_est[log_index]  = tau_est.A1
    log_tau_des[log_index]  = tau_des.A1
    log_t[log_index]        = t
    if not log_index%100 :
        print "com error \t{0}".format(np.linalg.norm(tsid.data.comerr))
    return np.vstack([f_disturb_traj(t),tau_ctrl])
    
def torque_controller(tau_des,tau_est):
    return tau_des + Ktau*(tau_des-tau_est)
    
def f_disturb_traj(t):
     if (t>0.5 and t<0.8 ):
        return np.matrix([30.,0.,0.]).T
     if (t>0.9 and t<0.91 ):
        return np.matrix([300.,0.,0.]).T
     return np.matrix([0.,0.,0.]).T
    
def com_traj(t):
    start_position_y,stop_position_y = 0.0 , 0.07
    start_position_z,stop_position_z = 0.53 , 0.53
    travel_time = 2.0
    py,vy,ay = traj_sinusoid(t,start_position_y,stop_position_y,travel_time)
    pz,vz,az = traj_sinusoid(t,start_position_z,stop_position_z,travel_time)
    return np.matrix([[py ],[pz]]) , np.matrix([[vy ],[vz]]) , np.matrix([[ay ],[az]])

def compute_torques_from_dv_and_forces(dv,f):
    M  = robot.data.M        #(7,7)
    h  = robot.data.nle      #(7,1)
    Jl = se3.frameJacobian(robot.model,robot.data,LF)       
    Jr = se3.frameJacobian(robot.model,robot.data,RF)
    Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
    tau = (M*dv - Jc.T*f + h)[3:]
    return tau

    
np.set_printoptions(precision=3)
np.set_printoptions(linewidth=200)
q = robot.q0.copy()
v = zero(NV)

#just integrate tsid dv:
#~ dt=simu.dt
#~ for i in range(log_size):
    #~ dv=stab.tsid(q,v,'dv')
    #~ v += dv*dt
    #~ q = se3.integrate(robot.model,q,v*dt)
    #~ robot.display(q)
    #~ time.sleep(0.001)


tsid.callback_com=com_traj


#~ def tsid_compact(q,v):
    #~ '''this call tsid to get forces and accelerations, then return torques from rnea'''
    #~ f=stab.tsid(q,v,'f')
    #~ lk_flf = se3.Force(np.vstack([0,f[:2],0,0,0]))
    #~ rk_frf = se3.Force(np.vstack([0,f[2:],0,0,0]))
    #~ forces_full = ForceDict({ simu.RK: rk_frf, LK: lk_flf},simu.NB)
    #~ return se3.rnea(robot.model,robot.data,q,v,stab.tsid(q,v,'dv'),forces_full)

#~ print "tau via rnea and external forces:"
#~ print tsid_compact(q,v)
#~ print "tau via tsid"
#~ print stab.tsid(q,v,'tau')
f=np.matrix([0.,0.,0.,0.]).T
#control the robot with an inverse dynamic:
print "looping on TSID"
v[:] = 0
#~ control = stab.tsid
control = controller
q,v = loop(q,v,f,log_size)

ax1 = plt.subplot(311)
infostr = "Infos:"
infostr += "\n Ktau  = {}".format(Ktau)
infostr += "\n fc FTfilter = {} Hz".format(fc)
infostr += "\n Kp_post {}".format(Kp_post)
infostr += "\n Kp_com {}".format(Kp_com)
infostr += "\n tauc {}".format(tauc)
infostr += "\n Ky={} Kz={}".format(Ky,Kz)
infostr += "\n dt={}ms ndt={}".format(dt*1000,ndt)

plt.text(0.1,0.05,infostr,
        bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
plt.title('com tracking Y')
plt.plot(log_t,log_com[:,0],    label="com")
plt.plot(log_t,log_comerr[:,0], label="com error")
plt.plot(log_t,log_comref[:,0], label="com ref")
plt.legend()
plt.subplot(312,sharex=ax1)
plt.title('feet forces Y') 
plt.plot(log_t,log_lkf_sensor[:,0],label="applied force",color='b')
plt.plot(log_t,log_rkf_sensor[:,0],label="applied force",color='r')
plt.plot(log_t,log_lkf[:,0], label="force TSID",color='b',linestyle=':')
plt.plot(log_t,log_rkf[:,0], label="force TSID",color='r',linestyle=':')
plt.legend()
plt.subplot(313,sharex=ax1)
plt.title('feet forces Z')
plt.plot(log_t,log_lkf_sensor[:,1],label="applied force",color='b')
plt.plot(log_t,log_rkf_sensor[:,1],label="applied force",color='r')
plt.plot(log_t,log_lkf[:,1], label="force TSID",color='b',linestyle=':')
plt.plot(log_t,log_rkf[:,1], label="force TSID",color='r',linestyle=':')
plt.legend()
plt.show()
embed()


