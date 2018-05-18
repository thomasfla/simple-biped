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
from tsid_flexible_contacts import TsidFlexibleContact
from path import pkg, urdf 
from noise_utils import NoisyState
useViewer = True
np.set_printoptions(precision=3)
np.set_printoptions(linewidth=200)

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
   
#Plots
PLOT_COM_AND_FORCES = True
PLOT_COM_DERIVATIVES = True   
PLOT_ANGULAR_MOMENTUM_DERIVATIVES = True   
   
#Simulation parameters
dt  = 1e-3
ndt = 1
simulation_time = 2.0

#robot parameters
tauc = 0*np.array([1.,1.,1.,1.])#coulomb friction
Ky = 23770.
Kz = 239018.
By = 50. *0
Bz = 500.*0
Kspring = -np.diagflat([Ky,Kz,0.])     # Stiffness of the feet spring
Bspring = -np.diagflat([By,Bz,0.])     # damping of the feet spring

#Controller parameters
FLEXIBLE_CONTROLLER = True
DISTURB = False
fc      = np.inf #cutoff frequency of the Force fiter
Ktau    = 0.0
Kp_post = 10
Kp_com  = 30
w_post = 1
FTSfilter = FIR1LowPass(np.exp(-2*np.pi*fc*dt)) #Force sensor filter
#~ b, a = np.array ([0.00554272,  0.01108543,  0.00554272]), np.array([1., -1.77863178,  0.80080265])
#~ FTSfilter = BALowPass(b,a,"butter_lp_filter_Wn_05_N_2")  

log_size = int(simulation_time / dt)    #max simulation samples
log_comref = np.zeros([log_size,2])+np.nan    #Centrer of mass reference
log_com_p_err = np.zeros([log_size,2])+np.nan #Centrer of mass error
log_com_v_err = np.zeros([log_size,2])+np.nan #Centrer of mass velocity error
log_com_a_err = np.zeros([log_size,2])+np.nan #Centrer of mass acceleration error
log_com_j_err = np.zeros([log_size,2])+np.nan #Centrer of mass jerk error
log_com_s_des = np.zeros([log_size,2])+np.nan #Desired centrer of mass snap
log_com_p     = np.zeros([log_size,2])+np.nan #Centrer of mass
log_real_com_p= np.zeros([log_size,2])+np.nan #Centrer of mass computed with real state
log_com_v     = np.zeros([log_size,2])+np.nan #Centrer of mass velocity
log_real_com_v= np.zeros([log_size,2])+np.nan #Centrer of mass velocity with real state
log_com_a     = np.zeros([log_size,2])+np.nan #Centrer of mass acceleration
log_com_j     = np.zeros([log_size,2])+np.nan #Centrer of mass jerk
log_iam       = np.zeros([log_size,1])+np.nan #Integral of the Angular Momentum approximated by the base orientation
log_am        = np.zeros([log_size,1])+np.nan #Angular Momentum
log_dam       = np.zeros([log_size,1])+np.nan #Angular Momentum derivative
log_ddam      = np.zeros([log_size,1])+np.nan #Angular Momentum 2nd derivative
log_dddam_des = np.zeros([log_size,1])+np.nan #Desired Angular Momentum 3rd derivative
log_lkf       = np.zeros([log_size,2])+np.nan #left  knee force
log_rkf       = np.zeros([log_size,2])+np.nan #right knee force
log_lkf_sensor = np.zeros([log_size,2])+np.nan  #left  knee force from FT/sensor (from deformation)
log_rkf_sensor = np.zeros([log_size,2])+np.nan  #right knee force from FT/sensor (from deformation)
log_lkdf_sensor = np.zeros([log_size,2])+np.nan #left  knee force from FT/sensor (from deformation velocity)
log_rkdf_sensor = np.zeros([log_size,2])+np.nan #right knee force from FT/sensor (from deformation velocity)
log_tau_des = np.zeros([log_size,4])+np.nan   #torques desired by TSID
log_tau_est = np.zeros([log_size,4])+np.nan   #torques estimated by contact forces
log_tau_ctrl = np.zeros([log_size,4])+np.nan  #torque sent to the motors
log_a_lf_fd  = np.zeros([log_size,3])+np.nan  #left foot acceleration via finite differences
log_a_rf_fd  = np.zeros([log_size,3])+np.nan  #right foot acceleration via finite differences
log_a_lf_jac  = np.zeros([log_size,3])+np.nan #left foot acceleration via jacobian formula
log_a_rf_jac  = np.zeros([log_size,3])+np.nan #right foot acceleration via jacobian formula
log_a_lf_des  = np.zeros([log_size,2])+np.nan #left foot acceleration desired
log_a_rf_des = np.zeros([log_size,2])+np.nan  #right foot acceleration desired
log_q        = np.zeros([log_size,8])+np.nan  #conf vector
log_v_lf     = np.zeros([log_size,2])+np.nan  #left foot velocity
log_v_rf     = np.zeros([log_size,2])+np.nan  #right foot velocity
log_p_lf     = np.zeros([log_size,2])+np.nan  #left foot position
log_p_rf     = np.zeros([log_size,2])+np.nan  #right foot position
log_dv_simu   = np.zeros([log_size,7])+np.nan #dv from simulator
log_dv_tsid   = np.zeros([log_size,7])+np.nan #dv desired from TSID
log_robotInertia = np.zeros([log_size,1])+np.nan #robot inertia
last_vlf = np.matrix([0,0,0]).T;
last_vrf = np.matrix([0,0,0]).T;

log_t        = np.zeros([log_size,1])+np.nan  # time
log_index = 0  

simu = Simu(robot,dt=dt,ndt=ndt)
simu.tauc = tauc
simu.Krf = Kspring
simu.Klf = Kspring
simu.Brf = Bspring
simu.Blf = Bspring

if FLEXIBLE_CONTROLLER:
    #~ Kp_com = 1
    #~ Kd_com = 2.7
    #~ Ka_com = 3.4
    #~ Kj_com = 2.1
    #~ Kp_post = 1e-6
    
    Kp_com = 1.20e+06
    Kd_com = 1.54e+05
    Ka_com = 7.10e+03
    Kj_com = 1.40e+02
    #~ Kp_com,Kd_com,Ka_com,Kj_com = 2.4e+09, 5.0e+07, 3.5e+05, 1.0e+03
    #~ Kp_com,Kd_com,Ka_com,Kj_com = 17160.0,  6026.0,   791.0,    46.  
    Kp_post = 10
    w_post  = 0.1
    tsid=TsidFlexibleContact(robot,Ky,Kz,w_post,Kp_post,Kp_com, Kd_com, Ka_com, Kj_com)
else:
    tsid=Tsid(robot,Kp_post,Kp_com)
NQ,NV,NB,RF,LF,RK,LK = simu.NQ,simu.NV,simu.NB,simu.RF,simu.LF,simu.RK,simu.LK
def loop(q,v,f,df,niter,ndt=None,dt=None,tsleep=.9,fdisplay=100):
    global log_index
    t0 = time.time()
    if dt  is not None: simu.dt  = dt
    if ndt is not None: simu.ndt = ndt
    robot.display(q)
    for i in range(niter):
        log_index = i
        # add noise to the perfect state q,v,f,df
        q_noisy,v_noisy,f_noisy,df_noisy = ns.get_noisy_state(q,v,f,df)
        q,v,f,df = simu(q,v,control(q_noisy,v_noisy,f_noisy,df_noisy))
        #~ q,v,f,df = simu(q,v,control(q,v,f,df))  
        #log the real com and his derivatives
        se3.centerOfMass(robot.model,robot.data,q,v,zero(NV))
        log_real_com_p[log_index]   = robot.data.com[0][1:].A1
        log_real_com_v[log_index]   = robot.data.vcom[0][1:].A1
        
        com_v = robot.data.vcom[0][1:]
        
        log_lkf_sensor[log_index] =f[:2].A1
        log_rkf_sensor[log_index] =f[2:].A1
        log_lkdf_sensor[log_index] =df[:2].A1
        log_rkdf_sensor[log_index] =df[2:].A1
        log_dv_simu[log_index] =simu.dv.A1
        #get feet accelerations via finite differences
        global last_vlf,last_vrf
        log_a_lf_fd[log_index] = (simu.vlf-last_vlf).A1/simu.dt
        log_a_rf_fd[log_index] = (simu.vrf-last_vrf).A1/simu.dt
        last_vlf = simu.vlf
        last_vrf = simu.vrf
        #get feet acceleration via jacobian.
        Jl,Jr = robot.get_Jl_Jr_world(q)
        driftLF,driftRF = robot.get_driftLF_driftRF_world(q,v)
        log_a_lf_jac[log_index] = (driftLF.vector + Jl*simu.dv)[1:4].A1
        log_a_rf_jac[log_index] = (driftRF.vector + Jr*simu.dv)[1:4].A1
        Mlf,Mrf = robot.get_Mlf_Mrf(q)
        log_p_lf[log_index] = Mlf.translation[1:3].A1
        log_p_rf[log_index] = Mrf.translation[1:3].A1
        log_v_lf[log_index] = simu.vlf[:2].A1 # Ty, Tz, not Rx
        log_v_rf[log_index] = simu.vrf[:2].A1
        if not i % fdisplay:
            robot.display(q)
            while((time.time()-t0)<(i*simu.dt)):
                time.sleep(0.01*simu.dt) # 1% jitter
    print 'Elapsed time = %.2f (simu time=%.2f)' % (time.time()-t0,simu.dt*niter)
    return q,v

def traj_sinusoid(t,start_position,stop_position,travel_time):
    # a cos(bt) + c
    A=-(stop_position-start_position)*0.5
    B = np.pi/travel_time
    C = start_position+(stop_position-start_position)*0.5
    
    p =         A*np.cos(B*t) + C
    v =      -B*A*np.sin(B*t)
    a =    -B*B*A*np.cos(B*t)
    j =   B*B*B*A*np.sin(B*t)
    s = B*B*B*B*A*np.cos(B*t) 
    return p,v,a,j,s
# check derivatives
eps = 1e-8
p1,v1,a1,j1,s1 =  traj_sinusoid(0.123   ,1.1,2.2,3.3)
p2,v2,a2,j2,s2 =  traj_sinusoid(0.123+eps,1.1,2.2,3.3)
assert isapprox(v1,(p2-p1)/eps)
assert isapprox(a1,(v2-v1)/eps)
assert isapprox(j1,(a2-a1)/eps)
assert isapprox(s1,(j2-j1)/eps)

def controller(q,v,f,df):
    ''' Take the sensor data (position, velocity and contact forces) 
    and generate a torque command'''
    t=log_index*simu.dt
    #filter forces
    f_filtered = FTSfilter.update(f)
    if FLEXIBLE_CONTROLLER:
        tsid.solve(q,v,f,df,t)
    else:
        tsid.solve(q,v,t)
    dv      = tsid.data.dv
    tau_des = tsid.data.tau
    log_q[log_index] = q.A1
    log_com_p_err[log_index] = tsid.data.com_p_err
    log_com_v_err[log_index] = tsid.data.com_v_err
    log_com_p[log_index] = tsid.data.com_p
    log_com_v[log_index] = tsid.data.com_v
    
    log_iam[log_index]       = tsid.data.iam
    log_am[log_index]        = tsid.data.am
    log_dam[log_index]       = tsid.data.dam
    log_ddam[log_index]      = tsid.data.ddam
    log_dddam_des[log_index] = tsid.data.dddam_des
    
    if FLEXIBLE_CONTROLLER:
        log_com_a_err[log_index] = tsid.data.com_a_err
        log_com_j_err[log_index] = tsid.data.com_j_err
        log_com_a[log_index] = tsid.data.com_a
        log_com_j[log_index] = tsid.data.com_j
        #get com snap via finite fifferences
        global last_com_j
        log_com_s_des[log_index] = tsid.data.com_s_des
        last_com_j = tsid.data.com_j
    log_comref[log_index]    = tsid.data.comref
    log_lkf[log_index]       = tsid.data.lkf
    log_rkf[log_index]       = tsid.data.rkf
    log_dv_tsid[log_index]   = tsid.data.dv.A1
    log_a_lf_des[log_index]  = tsid.data.lf_a_des.A1
    log_a_rf_des[log_index]  = tsid.data.rf_a_des.A1
    log_robotInertia[log_index] = tsid.data.robotInertia
    #estimate actual torque from contact forces
    tau_est  = compute_torques_from_dv_and_forces(dv,f_filtered)
    tau_ctrl = torque_controller(tau_des,tau_est)
    log_tau_ctrl[log_index] = tau_ctrl.A1
    log_tau_est[log_index]  = tau_est.A1
    log_tau_des[log_index]  = tau_des.A1
    log_t[log_index]        = t
    if not log_index%100 :
        print "t:{0} \t com error \t{1} ".format(log_index*dt, np.linalg.norm(tsid.data.com_p_err))
    return np.vstack([f_disturb_traj(t),tau_ctrl])
def torque_controller(tau_des,tau_est):
    return tau_des + Ktau*(tau_des-tau_est)
    
def f_disturb_traj(t):
    if DISTURB:
        if (t>0.5 and t<0.8 ):
            return np.matrix([30.,0.,0.]).T
        #~ if (t>0.9 and t<0.91 ):
            #~ return np.matrix([300.,0.,0.]).T
    return np.matrix([0.,0.,0.]).T
    
def com_traj(t):
    start_position_y,stop_position_y = 0.0 , 0.07
    start_position_z,stop_position_z = 0.53 , 0.53
    travel_time = 2.0
    py,vy,ay,jy,sy = traj_sinusoid(t,start_position_y,stop_position_y,travel_time)
    pz,vz,az,jz,sz = traj_sinusoid(t,start_position_z,stop_position_z,travel_time)
    return (np.matrix([[py ],[pz]]), 
            np.matrix([[vy ],[vz]]), 
            np.matrix([[ay ],[az]]), 
            np.matrix([[jy ],[jz]]), 
            np.matrix([[sy ],[sz]]))

def com_const(t):
    return (np.matrix([[0 ],[0.53]]) , 
            np.matrix([[0],[0]]), 
            np.matrix([[0],[0]]), 
            np.matrix([[0],[0]]), 
            np.matrix([[0],[0]]))
    
def compute_torques_from_dv_and_forces(dv,f):
    M  = robot.data.M        #(7,7)
    h  = robot.data.nle      #(7,1)
    Jl = se3.frameJacobian(robot.model,robot.data,LF)       
    Jr = se3.frameJacobian(robot.model,robot.data,RF)
    Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
    tau = (M*dv - Jc.T*f + h)[3:]
    return tau

tsid.callback_com=com_traj
#~ tsid.callback_com=com_const

q = robot.q0.copy()
v = zero(NV)
g=9.81
v[:] = 0
se3.computeAllTerms(robot.model,robot.data,robot.q0,v)
m = robot.data.mass[0] 
#initial state
q[1]-=0.5*m*g/Kz
f,df = simu.compute_f_df_from_q_v(q,v)


#control the robot with an inverse dynamic:
print "looping on TSID"

#~ control = stab.tsid
ns = NoisyState(dt)
control = controller
q,v = loop(q,v,f,df,log_size)

if PLOT_COM_AND_FORCES:
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
    plt.plot(log_t,log_com_p[:,0],    label="com")
    plt.plot(log_t,log_com_p_err[:,0], label="com error")
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


def finite_diff(data,dt):
    fd = data.copy()
    fd[0] = 0.
    fd[1:,] -= data[:-1,]
    fd = fd *(1/dt)
    fd[0] = np.nan # just ignore the first points for display
    return fd
    

if PLOT_COM_DERIVATIVES :
    axe_label = {0:'Y', 1:'Z'}    
    for i in [0,1]:
        ax1 = plt.subplot(511)
        plt.plot(log_t,log_com_p[:,i], label="com " + axe_label[i]) 
        plt.plot(log_t,log_real_com_p[:,i], label="com real" + axe_label[i]) 
        plt.legend()
        plt.subplot(512,sharex=ax1)
        plt.plot(log_t,log_com_v[:,i], label="vcom "+ axe_label[i]) 
        plt.plot(log_t,log_real_com_v[:,i], label="vcom real"+ axe_label[i]) 
        plt.plot(log_t,finite_diff(log_com_p[:,i],dt),':', label="fd com " + axe_label[i]) 
        plt.legend()
        plt.subplot(513,sharex=ax1)
        plt.plot(log_t,log_com_a[:,i], label="acom "+ axe_label[i]) 
        plt.plot(log_t,finite_diff(log_com_v[:,i],dt),':', label="fd vcom " + axe_label[i]) 
        plt.legend()
        plt.subplot(514,sharex=ax1)
        plt.plot(log_t,log_com_j[:,i], label="jcom "+ axe_label[i]) 
        plt.plot(log_t,finite_diff(log_com_a[:,i],dt),':', label="fd acom " + axe_label[i]) 
        plt.legend()
        plt.subplot(515,sharex=ax1)
        plt.plot(log_t,log_com_s_des[:,i], label="desired scom" + axe_label[i])
        plt.plot(log_t,finite_diff(log_com_j[:,i],dt),':', label="fd scom " + axe_label[i])
        #~ plt.plot(log_t,log_lkf_sensor[:,1], label="force Left z")
        #~ plt.plot(log_t,log_rkf_sensor[:,1], label="force Right z")
        plt.legend()
        plt.show()

if PLOT_ANGULAR_MOMENTUM_DERIVATIVES:
    ax1 = plt.subplot(511)
    plt.plot(log_t,log_iam, label="iam ") 
    plt.legend()
    plt.subplot(512,sharex=ax1)
    plt.plot(log_t,log_am, label="am ") 
    plt.plot(log_t,finite_diff(log_iam,dt),':', label="fd iam ") 
    plt.legend()
    plt.subplot(513,sharex=ax1)
    plt.plot(log_t,log_dam, label="dam ") 
    plt.plot(log_t,finite_diff(log_am,dt),':', label="fd am " ) 
    plt.legend()
    plt.subplot(514,sharex=ax1)
    plt.plot(log_t,log_ddam, label="ddam ") 
    plt.plot(log_t,finite_diff(log_dam,dt),':', label="fd dam ") 
    plt.legend()
    plt.subplot(515,sharex=ax1)
    plt.plot(log_t,log_dddam_des, label="desired dddam")
    plt.plot(log_t,finite_diff(log_ddam,dt),':', label="fd ddam")

plt.legend()
plt.show()

#~ plt.plot(log_a_lf_fd, label = "a_lf_fd")
#~ plt.plot(log_a_rf_fd, label = "a_rf_fd")
#~ plt.plot(log_a_lf_jac, label = "a_lf_jac")
#~ plt.plot(log_a_rf_jac, label = "a_rf_jac")

log_a_lf_fd[0]=np.nan
log_a_rf_fd[0]=np.nan
plt.plot(log_a_lf_fd[:,:2], label = "log_a_lf_fd")
plt.plot(log_a_lf_jac[:,:2], label = "a_lf_jac")
plt.plot(log_a_lf_des, label = "log_a_lf_des")

plt.legend()
plt.show()

plt.plot(log_dv_simu[:] - log_dv_tsid[:], label = "dv_simu-dv_tsid")
plt.legend()
plt.show()

plt.plot(log_t,log_v_lf,label="lf vel" + axe_label[i])
plt.plot(log_t,log_v_rf,label="rf vel" + axe_label[i])
plt.plot(log_t,finite_diff(log_p_lf,dt),':',lw=2, label="fd lf pos")
plt.plot(log_t,finite_diff(log_p_rf,dt),':',lw=2, label="fd rf pos")
plt.legend()
plt.show()

#~ plt.plot(log_t,log_robotInertia)
#~ plt.show()

embed()



