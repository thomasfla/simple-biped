import pinocchio as se3
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import *
from math import pi,sqrt,cos,sin
from hrp2_reduced import Hrp2Reduced
import time 
from simu import Simu, ForceDict
from quadprog import solve_qp
from utils_thomas import restert_viewer_server
from logger import Logger
from filters import FIR1LowPass, BALowPass, FiniteDiff
import matplotlib.pyplot as plt
import quadprog
from tsid import Tsid
from tsid_flexible_contacts import TsidFlexibleContact
from path import pkg, urdf 
from noise_utils import NoisyState
from estimators import Kalman, get_com_and_derivatives

from estimation.momentumEKF import *
from plot import plot_gain_stability
import os
import time

try:
    from IPython import embed
except ImportError:
    pass

useViewer = False
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
PLOT_ANGULAR_MOMENTUM_DERIVATIVES = False   
   
#Simulation parameters
dt  = 1e-3
ndt = 1
simulation_time = 20.0
USE_REAL_STATE = True       # use real state for controller feedback
#robot parameters
tauc = 0.*np.array([1.,1.,1.,1.])#coulomb friction
Ky = 23770.
Kz = 239018.
By = 50. *0.
Bz = 500.*0.
Kspring = -np.diagflat([Ky,Kz,0.])     # Stiffness of the feet spring
Bspring = -np.diagflat([By,Bz,0.])     # damping of the feet spring

#Controller parameters
fc_dtau_filter = 100.           # cutoff frequency of the filter applyed to the finite differences of the torques 
FLEXIBLE_CONTROLLER = True      # if True it uses the controller for flexible contacts
DISTURB = False                 # if True disturb the motion with an external force
fc      = np.inf                # cutoff frequency of the Force fiter
Ktau    = 2.0                   # torque proportional feedback gain
Kdtau   = 2.*sqrt(Ktau)*0.00    # Make it unstable ??
Kp_post = 10                    # postural task proportional feedback gain
Kp_com  = 30                    # com proportional feedback gain
Kd_com = 2*sqrt(Kp_com)         # com derivative feedback gain
if(FLEXIBLE_CONTROLLER):
    w_post  = 0.1
else:
    w_post = 0.001                  # postural task weight
FTSfilter = FIR1LowPass(np.exp(-2*np.pi*fc*dt)) #Force sensor filter

#Grid of gains to try:
#Kd_coms = np.linspace(1,100,50)
#Kp_coms = np.linspace(1,500,50)
Kp_coms = [1,1]
Kd_coms = [1,100]

#Simulator
simu = Simu(robot,dt=dt,ndt=ndt)
simu.tauc = tauc
simu.Krf = Kspring
simu.Klf = Kspring
simu.Brf = Bspring
simu.Blf = Bspring
# size of configuration vector (NQ), velocity vector (NV), number of bodies (NB)
NQ,NV,NB,RF,LF,RK,LK = simu.NQ,simu.NV,simu.NB,simu.RF,simu.LF,simu.RK,simu.LK

#initial state
q0 = robot.q0.copy()
v0 = zero(NV)
g=9.81
se3.computeAllTerms(robot.model,robot.data,robot.q0,v0)
m = robot.data.mass[0] 
q0[1]-=0.5*m*g/Kz
f0,df0 = simu.compute_f_df_from_q_v(q0,v0)
c0,dc0,ddc0,dddc0 = get_com_and_derivatives(robot,q0,v0,f0,df0)
l0 = 0


dtau_fd_filter = FiniteDiff(dt)
dtau_lp_filter = FIR1LowPass(np.exp(-2*np.pi*fc_dtau_filter*dt)) #Force sensor filter


#Noise applied on the state to get a simulated measurement
ns = NoisyState(dt,robot,Ky,Kz)

# noise standard deviation
n_x = 9+4
n_u = 4
n_y = 9
sigma_x_0 = 1e0              # initial state estimate std dev
sigma_ddf = 1e2*ones(4)      # control (i.e. force accelerations) noise std dev used in EKF
sigma_ddf_sim = 1e2*ones(4)  # control (i.e. force accelerations) noise std dev used in simulation
sigma_c  = 1e-3*ones(2)      # CoM position measurement noise std dev
sigma_dc = 1e-2*ones(2)      # CoM velocity measurement noise std dev
sigma_l  = 1e0*ones(1)      # angular momentum measurement noise std dev
sigma_f  = 1e-2*m*ones(4)       # force measurement noise std dev
S_0 = sigma_x_0**2 * np.eye(n_x)


#to be replaced with EKF momentum estimator
stddev_p = 0.001 
stddev_v = ns.std_gyry #gyro noise
stddev_a = ns.std_fz / m #fts noise / m
estimator = Kalman(dt, stddev_p,stddev_v,stddev_a, 1e2, c0,dc0,ddc0,dddc0)

#andrea's estimator works with array...
c0_array  = c0.A1
dc0_array = dc0.A1
dc0_array = dc0.A1
l0_array  = np.array([l0])
f0_array = f0.A1
centroidalEstimator = MomentumEKF(dt, m, g, c0_array, dc0_array, l0_array, f0_array, S_0, sigma_c, sigma_dc, sigma_l, sigma_f, sigma_ddf)

#~ estimator = None
#~ b, a = np.array ([0.00554272,  0.01108543,  0.00554272]), np.array([1., -1.77863178,  0.80080265])
#~ FTSfilter = BALowPass(b,a,"butter_lp_filter_Wn_05_N_2")  

log_size = int(simulation_time / dt)    #max simulation samples
log_comref = np.zeros([log_size,2])+np.nan    #Centrer of mass reference
log_com_p_err = np.zeros([log_size,2])+np.nan #Centrer of mass error
log_com_v_err = np.zeros([log_size,2])+np.nan #Centrer of mass velocity error
log_com_a_err = np.zeros([log_size,2])+np.nan #Centrer of mass acceleration error
log_com_j_err = np.zeros([log_size,2])+np.nan #Centrer of mass jerk error

if FLEXIBLE_CONTROLLER:
    log_com_s_des = np.zeros([log_size,2])+np.nan #Desired centrer of mass snap
else:
    log_com_a_des = np.zeros([log_size,2])+np.nan #Desired centrer of acceleration

log_com_p_mes = np.zeros([log_size,2])+np.nan #Measured Centrer of mass
log_com_v_mes = np.zeros([log_size,2])+np.nan #Measured Centrer of mass velocity
log_com_a_mes = np.zeros([log_size,2])+np.nan #Measured Centrer of mass acceleration

log_com_p_est = np.zeros([log_size,2])+np.nan #Estimated Centrer of mass
log_com_v_est = np.zeros([log_size,2])+np.nan #Estimated Centrer of mass velocity
log_com_a_est = np.zeros([log_size,2])+np.nan #Estimated Centrer of mass acceleration
log_com_j_est = np.zeros([log_size,2])+np.nan #Estimated Centrer of mass jerk

log_com_p     = np.zeros([log_size,2])+np.nan #Centrer of mass computed with real state
log_com_v     = np.zeros([log_size,2])+np.nan #Centrer of mass velocity computed with real state
log_com_a     = np.zeros([log_size,2])+np.nan #Centrer of mass acceleration computed with real state
log_com_j     = np.zeros([log_size,2])+np.nan #Centrer of mass jerk computed with real state

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
log_tau_est_fd = np.zeros([log_size,4])+np.nan   #derivative of tau_est by finite differences
log_dtau_est = np.zeros([log_size,4])+np.nan   #filtered derivative of tau_est by finite differences
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
    tsid=TsidFlexibleContact(robot,Ky,Kz,w_post,Kp_post,Kp_com, Kd_com, Ka_com, Kj_com, estimator)
else:
    tsid=Tsid(robot,Ky,Kz,Kp_post,Kp_com,w_post)

def loop(q,v,f,df,niter,ndt=None,dt=None,tsleep=.9,fdisplay=100):
    global log_index
    t0 = time.time()
    if dt  is not None: simu.dt  = dt
    if ndt is not None: simu.ndt = ndt
    robot.display(q)
    for i in range(niter):
        log_index = i
        # add noise to the perfect state q,v,f,df
        if USE_REAL_STATE:
            q_noisy,v_noisy,f_noisy,df_noisy = q,v,f,df
        else:
            q_noisy,v_noisy,f_noisy,df_noisy = ns.get_noisy_state(q,v,f,df)
        
        #Run the centroidal estimation (not used yet)
        com_noisy, com_v_noisy, com_a_noisy, com_j_noisy = get_com_and_derivatives(robot,q_noisy,v_noisy,f_noisy,df_noisy)
        l_noisy = robot.get_angularMomentum(q_noisy,v_noisy)
         #convert
        com_noisy_array = com_noisy.A1
        com_v_noisy_array = com_v_noisy.A1
        l_noisy_array = np.array([l_noisy])
        #~ centroidalEstimator.predict_update(com_noisy, com_v_noisy_array, l_noisy_array, f_noisy, p, ddf)
        
        #Run controller
        u = control(q_noisy,v_noisy,f_noisy,df_noisy)
        
        #simulate the system
        q,v,f,df = simu(q,v,u)
        
        #log the real com and his derivatives
        com, com_v, com_a, com_j = get_com_and_derivatives(robot,q,v,f,df)
        log_com_p[log_index]   = com.A1
        log_com_v[log_index]   = com_v.A1
        log_com_a[log_index]   = com_a.A1
        log_com_j[log_index]   = com_j.A1

        log_lkf_sensor[log_index] =f[:2].A1 #todo rename
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
#eps = 1e-8
#p1,v1,a1,j1,s1 =  traj_sinusoid(0.123   ,1.1,2.2,3.3)
#p2,v2,a2,j2,s2 =  traj_sinusoid(0.123+eps,1.1,2.2,3.3)
#assert isapprox(v1,(p2-p1)/eps)
#assert isapprox(a1,(v2-v1)/eps)
#assert isapprox(j1,(a2-a1)/eps)
#assert isapprox(s1,(j2-j1)/eps)

def controller(q,v,f,df):
    ''' Take the sensor data (position, velocity and contact forces) 
    and generate a torque command '''
    t=log_index*simu.dt
    #filter forces
    f_filtered = FTSfilter.update(f)
    tsid.solve(q,v,f,df,t)

    dv      = tsid.data.dv
    tau_des = tsid.data.tau
    log_q[log_index] = q.A1
    log_com_p_err[log_index] = tsid.data.com_p_err
    log_com_v_err[log_index] = tsid.data.com_v_err
    
    log_com_p_mes[log_index] = tsid.data.com_p_mes
    log_com_v_mes[log_index] = tsid.data.com_v_mes
    
    log_com_p_est[log_index] = tsid.data.com_p_est
    log_com_v_est[log_index] = tsid.data.com_v_est
    
    log_iam[log_index]       = tsid.data.iam
    log_am[log_index]        = tsid.data.am
    log_dam[log_index]       = tsid.data.dam
    log_ddam[log_index]      = tsid.data.ddam
    log_dddam_des[log_index] = tsid.data.dddam_des
    
    if FLEXIBLE_CONTROLLER:
        log_com_a_err[log_index] = tsid.data.com_a_err
        log_com_j_err[log_index] = tsid.data.com_j_err
        log_com_a_mes[log_index] = tsid.data.com_a_mes
        
        log_com_a_est[log_index] = tsid.data.com_a_est
        log_com_j_est[log_index] = tsid.data.com_j_est
        
        #get com snap via finite fifferences
        global last_com_j
        log_com_s_des[log_index] = tsid.data.com_s_des
        last_com_j = log_com_j[log_index-1] #tsid.data.com_j
    else :
        log_com_a_des[log_index] = tsid.data.com_a_des
    log_comref[log_index]    = tsid.data.comref
    log_lkf[log_index]       = tsid.data.lkf
    log_rkf[log_index]       = tsid.data.rkf
    log_dv_tsid[log_index]   = tsid.data.dv.A1
    log_a_lf_des[log_index]  = tsid.data.lf_a_des.A1
    log_a_rf_des[log_index]  = tsid.data.rf_a_des.A1
    log_robotInertia[log_index] = tsid.data.robotInertia
    #estimate actual torque from contact forces
    tau_est  = compute_torques_from_dv_and_forces(dv,f_filtered)
    tau_est_fd  = dtau_fd_filter.update(tau_est) 
    dtau_est = dtau_lp_filter.update(tau_est_fd)
    #~ tau_ctrl = torque_controller(tau_des,tau_est)

    if not FLEXIBLE_CONTROLLER:
        #~ tau_ctrl = tau_des + Ktau*(tau_des-tau_est) - Kdtau * dtau_est
        # regulation on forces
        #~ f_des = tsid.data.f 
        #~ f_ctrl = f_des + Ktau * Ktau * (f_des - f_filtered)
        #~ tau_ctrl = compute_torques_from_dv_and_forces(dv,f_ctrl) - Kdtau * df
        tau_ctrl = tau_des 
    else:
        tau_ctrl = tau_des 

    log_tau_ctrl[log_index]    = tau_ctrl.A1
    log_tau_est[log_index]     = tau_est.A1
    log_tau_est_fd[log_index]  = tau_est_fd.A1
    log_dtau_est[log_index]    = dtau_est.A1
    log_tau_des[log_index]     = tau_des.A1
    log_t[log_index]           = t
    if not log_index%100 :
        print "t:{0} \t com error \t{1} ".format(log_index*dt, np.linalg.norm(tsid.data.com_p_err))
    #check that the state does'nt go crazy
    if np.linalg.norm(tsid.data.com_p_err) > 0.1:
        raise ValueError("COM error > 0.1")
    if np.linalg.norm(tsid.data.f) > 350:
        raise ValueError("Forces > 350")
    if np.linalg.norm(q) > 10:
        raise ValueError("q > 10")
        
    return np.vstack([f_disturb_traj(t),tau_ctrl])
    
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
    Jl,Jr = robot.get_Jl_Jr_world(q, False)
    Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
    tau = (M*dv - Jc.T*f + h)[3:]
    return tau

#~ tsid.callback_com=com_traj
tsid.callback_com=com_const


#control the robot with an inverse dynamic:
control = controller
q,v,f,df = q0.copy(), v0.copy(), f0.copy(), df0.copy()
q,v = loop(q,v,f,df,log_size)
result = ""
KpKd = []
stab_grid = np.zeros([len(Kp_coms),len(Kd_coms)])+np.nan
Kp_grid = np.zeros([len(Kp_coms),len(Kd_coms)])+np.nan
Kd_grid = np.zeros([len(Kp_coms),len(Kd_coms)])+np.nan
i=0
for Kp_com in Kp_coms:
    j=0
    for Kd_com in Kd_coms:
        #change controller gains
        tsid.Kp_com = Kp_com
        tsid.Kd_com = Kd_com
        
        #reset all entity with internal states
        simu.reset()
        FTSfilter.reset()
        dtau_fd_filter.reset()
        dtau_lp_filter.reset()
        
        #start from initial state
        q,v,f,df = q0.copy(), v0.copy(), f0.copy(), df0.copy()
        isstable=True
        #simulate and test stability
        try:
            q,v = loop(q,v,f,df,log_size)
        except ValueError:
            isstable=False
        print "Kp_com={}, Kd_com={}, Stable? {}".format(Kp_com,Kd_com,isstable)
        KpKd.append([Kp_com,Kd_com])
        stab_grid[i,j] = isstable
        Kp_grid[i,j] = Kp_com
        Kd_grid[i,j] = Kd_com
        
        if isstable:
            result += "*"
        else:
            result += "-"
        print result
        j+=1
    result += "\n"
    i+=1
if i !=0:
    #save the stability region plot and data
    num = int(time.time()) #simple unique increasing timestamp
    outDir = "./data/{}/".format(num)
    os.makedirs(outDir)
    np.savez(outDir + "stab.npz", Kd_grid=Kd_grid, Kp_grid=Kp_grid, stab_grid=stab_grid)
    plot_gain_stability(Kd_grid,Kp_grid,stab_grid)
    plt.savefig(outDir + "stab.png")
    plt.show()

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
    plt.plot(log_t,log_com_p_mes[:,0],    label="measured com")
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
        plt.plot(log_t,log_com_p_mes[:,i], label="measured com " + axe_label[i]) 
        plt.plot(log_t,log_com_p_est[:,i], label="estimated com " + axe_label[i]) 
        plt.plot(log_t,log_com_p[:,i], label="com " + axe_label[i]) 
        plt.legend()
        plt.subplot(512,sharex=ax1)
        plt.plot(log_t,log_com_v_mes[:,i], label="measured vcom "+ axe_label[i]) 
        plt.plot(log_t,log_com_v_est[:,i], label="estimated vcom "+ axe_label[i]) 
        plt.plot(log_t,log_com_v[:,i], label="vcom "+ axe_label[i]) 
        plt.plot(log_t,finite_diff(log_com_p[:,i],dt),':', label="fd com " + axe_label[i]) 
        plt.legend()
        plt.subplot(513,sharex=ax1)
        plt.plot(log_t,log_com_a_mes[:,i], label="measured acom "+ axe_label[i]) 
        plt.plot(log_t,log_com_a_est[:,i], label="estimated acom "+ axe_label[i]) 
        if not FLEXIBLE_CONTROLLER :
            plt.plot(log_t,log_com_a_des[:,i], label="desired acom" + axe_label[i])
        plt.plot(log_t,log_com_a[:,i], label="acom "+ axe_label[i]) 
        plt.plot(log_t,finite_diff(log_com_v[:,i],dt),':', label="fd vcom " + axe_label[i]) 
        plt.legend()
        plt.subplot(514,sharex=ax1)
        plt.plot(log_t,log_com_j_est[:,i], label="estimated jcom "+ axe_label[i]) 
        plt.plot(log_t,log_com_j[:,i], label="jcom "+ axe_label[i]) 
        plt.plot(log_t,finite_diff(log_com_a[:,i],dt),':', label="fd acom " + axe_label[i])
        plt.legend()
        plt.subplot(515,sharex=ax1)
        if FLEXIBLE_CONTROLLER :
            plt.plot(log_t,log_com_s_des[:,i], label="desired scom" + axe_label[i])
        plt.plot(log_t,finite_diff(log_com_j[:,i],dt),':', label="fd jcom " + axe_label[i])
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
#~ 
plt.legend()
plt.show()

plt.plot(log_dv_simu[:] - log_dv_tsid[:], label = "dv_simu-dv_tsid")
plt.legend()
plt.show()

#~ plt.plot(log_t,log_v_lf,label="lf vel" + axe_label[i])
#~ plt.plot(log_t,log_v_rf,label="rf vel" + axe_label[i])
#~ plt.plot(log_t,finite_diff(log_p_lf,dt),':',lw=2, label="fd lf pos")
#~ plt.plot(log_t,finite_diff(log_p_rf,dt),':',lw=2, label="fd rf pos")
#~ plt.legend()
#~ plt.show()

#~ plt.plot(log_t,log_robotInertia)
#~ plt.show()


#~ plt.plot(log_t,log_tau_est, label="tau estimated via contact forces")
#~ plt.plot(log_t,log_tau_est_fd,label="finite differences of tau")
#~ plt.plot(log_t,1+log_dtau_est,label="filtered finite differences of tau")
#~ plt.legend()
#~ plt.show()
embed()
