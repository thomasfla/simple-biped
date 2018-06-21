import pinocchio as se3
from numpy import matlib
import numpy as np
from pinocchio.utils import *
from math import pi,sqrt
from hrp2_reduced import Hrp2Reduced
import time 
from simu import Simu
from utils.utils_thomas import restert_viewer_server, traj_sinusoid, finite_diff
from utils.logger import RaiLogger
from utils.filters import FIR1LowPass, FiniteDiff
import matplotlib.pyplot as plt
from utils.plot_utils import plot_from_logger
from tsid import Tsid
from tsid_flexible_contacts import TsidFlexibleContact
from path import pkg, urdf 
from utils.noise_utils import NoisyState
from estimation.momentumEKF import MomentumEKF

try:
    from IPython import embed
except ImportError:
    pass

useViewer = False
np.set_printoptions(precision=3, linewidth=200)

if useViewer:
    restert_viewer_server()
    
def controller(q,v,f):
    ''' Take the sensor data (position, velocity and contact forces) 
    and generate a torque command '''
    t=log_index*simu.dt
    tsid.solve(t, q, v, f)
    if not log_index%100 :
        print "t:%.1f \t com error \t%.3f " % (log_index*dt, np.linalg.norm(tsid.data.com_p_err))
    return np.vstack([f_disturb_traj(t), tsid.data.tau])
    
def f_disturb_traj(t):
    if (t>T_DISTURB_BEGIN and t<T_DISTURB_END ):
        return F_DISTURB
    return matlib.zeros(3).T
    
def com_traj(t, c_init, c_final, T):
    py,vy,ay,jy,sy = traj_sinusoid(t, c_init[0], c_final[0], T)
    pz,vz,az,jz,sz = traj_sinusoid(t, c_init[1], c_final[1], T)
    return (np.matrix([[py ],[pz]]), np.matrix([[vy ],[vz]]), 
            np.matrix([[ay ],[az]]), np.matrix([[jy ],[jz]]), np.matrix([[sy ],[sz]]))
    

robot = Hrp2Reduced(urdf,[pkg],loadModel=True,useViewer=useViewer)
robot.display(robot.q0)
if useViewer:
    robot.viewer.setCameraTransform(0,[1.9154722690582275, -0.2266872227191925, 0.1087859719991684,
                                       0.5243823528289795, 0.518651008605957, 0.4620114266872406, 0.4925136864185333])

#Simulation parameters
dt  = 1e-3
ndt = 5
simulation_time = 2.0

#robot parameters
tauc = 0.*np.array([1.,1.,1.,1.])#coulomb friction
Ky = 23770.
Kz = 239018.
By = 50e0
Bz = 500e0
Kspring = -np.diagflat([Ky,Kz,0.])     # Stiffness of the feet spring
Bspring = -np.diagflat([By,Bz,0.])     # damping of the feet spring

#Simulator
simu = Simu(robot,dt=dt,ndt=ndt)
simu.tauc = tauc
simu.Krf, simu.Klf = Kspring, Kspring
simu.Brf, simu.Blf = Bspring, Bspring
# size of configuration vector (NQ), velocity vector (NV), number of bodies (NB)
NQ,NV,NB,RF,LF,RK,LK = simu.NQ,simu.NV,simu.NB,simu.RF,simu.LF,simu.RK,simu.LK

#initial state
q0 = robot.q0.copy()
v0 = zero(robot.model.nv)
g_vec = robot.model.gravity.linear[1:].A1            # gravity acc vector
g = np.linalg.norm(g_vec)                            # gravity acceleration
se3.computeAllTerms(robot.model,robot.data,q0,v0)
m = robot.data.mass[0] 
q0[1]-=0.5*m*g/Kz
f0,df0 = simu.compute_f_df_from_q_v(q0,v0)
c0,dc0,ddc0,dddc0 = robot.get_com_and_derivatives(q0,v0,f0,df0)
l0 = 0

#Plots
PLOT_FORCES                         = 1
PLOT_COM_DERIVATIVES                = 1
PLOT_ANGULAR_MOMENTUM_DERIVATIVES   = 0   
PLOT_JOINT_TORQUES                  = 0   

USE_REAL_STATE = 0       # use real state for controller feedback
T_DISTURB_BEGIN = 0.50          # Time at which the disturbance starts
T_DISTURB_END   = 0.51          # Time at which the disturbance ends
F_DISTURB = np.matrix([0*5e2, 0, 0]).T

COM_REF_START = c0.A1 #[0.001,      0.527]
COM_REF_END   = c0.A1 + np.array([0.03, 0.0])
COM_TRAJ_TIME = 1.0

#Controller parameters
fc_dtau_filter = 100.           # cutoff frequency of the filter applyed to the finite differences of the torques 
CONTROLLER = 'tsid'             # either 'tsid' or 'tsid_flex' or 'tsid_adm'
fc      = np.inf                # cutoff frequency of the Force fiter
Ktau    = 2.0                   # torque proportional feedback gain
Kdtau   = 2.*sqrt(Ktau)*0.00    # Make it unstable ??
Kp_post = 10                    # postural task proportional feedback gain
if(CONTROLLER=='tsid_flex'):
    w_post  = 0.1
    (Kp_com, Kd_com, Ka_com, Kj_com) = (1.20e+06, 1.54e+05, 7.10e+03, 1.40e+02)
elif(CONTROLLER=='tsid'):
    w_post = 1e-2                  # postural task weight
    Kp_com = 30                    # com proportional feedback gain
    Kd_com = 2*sqrt(Kp_com)         # com derivative feedback gain
else:
    w_post = 0.001                  # postural task weight
    Kp_com = 30                    # com proportional feedback gain
    Kd_com = 2*sqrt(Kp_com)         # com derivative feedback gain

dtau_fd_filter = FiniteDiff(dt)
dtau_lp_filter = FIR1LowPass(np.exp(-2*pi*fc_dtau_filter*dt)) # Force sensor filter
FTSfilter      = FIR1LowPass(np.exp(-2*pi*fc*dt))             # Force sensor filter

#Noise applied on the state to get a simulated measurement
ns = NoisyState(dt,robot,Ky,Kz)

# noise standard deviation
n_x = 9+4
n_u = 4
n_y = 9
sigma_x_0 = 1e-2                    # initial state estimate std dev
sigma_ddf    = 1e4*np.ones(4)          # control (i.e. force accelerations) noise std dev used in EKF
sigma_f_dist = 1e1*np.ones(2)          # external force noise std dev used in EKF
sigma_c  = 1e-3*np.ones(2)             # CoM position measurement noise std dev
sigma_dc = ns.std_gyry*np.ones(2)      # CoM velocity measurement noise std dev
sigma_l  = 1e-1*np.ones(1)             # angular momentum measurement noise std dev
sigma_f  = np.array([ns.std_fy, ns.std_fz, ns.std_fy, ns.std_fz])  # force measurement noise std dev
S_0 = sigma_x_0**2 * np.eye(n_x)

centroidalEstimator = MomentumEKF(dt, m, g_vec, c0.A1, dc0.A1, np.array([l0]), f0.A1, S_0, sigma_c, sigma_dc, sigma_l, sigma_f, sigma_ddf, sigma_f_dist)

log_size = int(simulation_time / dt)    #max simulation samples
log_t        = np.zeros([log_size,1])+np.nan  # time
log_index = 0  

if CONTROLLER=='tsid_flex':
    tsid = TsidFlexibleContact(robot, Ky, Kz, w_post, Kp_post, Kp_com, Kd_com, Ka_com, Kj_com, centroidalEstimator)
else:
    tsid = Tsid(robot, Ky, Kz, Kp_post, Kp_com, w_post)

tsid.callback_com = lambda t : com_traj(t, COM_REF_START, COM_REF_END, COM_TRAJ_TIME)

# SETUP LOGGER
lgr = RaiLogger()
vc = 'vector'
vr = 'variable'
tsid_var_names  = ['dv', 'tau', 'com_p_err', 'com_v_err', 'com_p_mes', 'com_v_mes', 'com_p_est', 'com_v_est', 'comref', 'lkf', 'rkf']
tsid_var_types  = [ vc,     vc,     vc,          vc,          vc,            vc,           vc,         vc,       vc,      vc,    vc]
if CONTROLLER=='tsid_flex':
    tsid_var_names += [ 'lf_a_des', 'rf_a_des']
    tsid_var_types += [     vc,         vc    ]
    #Integral of angular momentum approximated by base orientation, angular momentum, its 1st and 2nd derivative, its desired 3rd derivative
    tsid_var_names += ['iam', 'am', 'dam', 'ddam', 'dddam_des', 'robotInertia']
    tsid_var_types += [  vr,   vr,    vr,     vr,       vr,          vr]
    tsid_var_names += ['com_a_err', 'com_j_err', 'com_a_mes', 'com_a_est', 'com_j_est', 'com_s_des']
    tsid_var_types += [     vc,          vc,          vc,          vc,          vc,            vc]
else:
    tsid_var_names += ['com_a_des']
    tsid_var_types += [     vc]
lgr.auto_log_variables(tsid.data, tsid_var_names, tsid_var_types, 'tsid')
lgr.auto_log_variables(simu, ['vlf', 'vrf', 'dv'], [vc, vc, vc], 'simu')
lgr.auto_log_variables(centroidalEstimator, ['x'], [vc], log_var_names=[['ekf_c_0', 'ekf_c_1', 'ekf_dc_0', 'ekf_dc_1', 'ekf_l', 'ekf_f_0', 'ekf_f_1',
                                                                         'ekf_f_2', 'ekf_f_3', 'ekf_df_0', 'ekf_df_1', 'ekf_df_2', 'ekf_df_3']])

lgr.auto_log_local_variables(['com_p', 'com_v', 'com_a', 'com_j', 'ddf'], [vc, vc, vc, vc, vc])
lgr.auto_log_local_variables(['f'], [vc], log_var_names=[['lkf_sensor_0', 'lkf_sensor_1', 'rkf_sensor_0', 'rkf_sensor_1']])
lgr.auto_log_local_variables(['df'], [vc], log_var_names=[['lkdf_sensor_0', 'lkdf_sensor_1', 'rkdf_sensor_0', 'rkdf_sensor_1']])
    
def loop(q, v, f, df, niter, ndt=None, dt=None, tsleep=.9, fdisplay=100):
    global log_index
    last_df  = np.matrix(np.zeros(4)).T
    t0 = time.time()
    if dt  is not None: simu.dt  = dt
    if ndt is not None: simu.ndt = ndt
    robot.display(q)
    
    com_p, com_v, com_a, com_j = robot.get_com_and_derivatives(q,v,f,df)
    print "com initial state:\n", com_p.T, com_v.T, com_a.T, com_j.T
    
    for i in range(niter):
        log_index = i
        # add noise to the perfect state q,v,f,df
        if USE_REAL_STATE:
            q_noisy,v_noisy,f_noisy,df_noisy = q,v,f,df
        else:
            q_noisy,v_noisy,f_noisy,df_noisy = ns.get_noisy_state(q,v,f,df)
        
        # simulate the system
        u = controller(q_noisy,v_noisy,f_noisy)
        q,v,f,df = simu(q,v,u)

        # log data        
        log_t[log_index] = log_index*simu.dt
        com_p, com_v, com_a, com_j = robot.get_com_and_derivatives(q,v,f,df)
        ddf = (df-last_df)/simu.dt;
        last_df = df
        
        lgr.log_all(locals())

        if not i % fdisplay:
            robot.display(q)
            while((time.time()-t0)<(i*simu.dt)):
                time.sleep(0.01*simu.dt) # 1% jitter
                
    print 'Elapsed time = %.2f (simu time=%.2f)' % (time.time()-t0, simu.dt*niter)
    return q,v

# control the robot with an inverse dynamic:
q,v,f,df = q0.copy(), v0.copy(), f0.copy(), df0.copy()
q,v = loop(q,v,f,df,log_size)

if PLOT_FORCES:
    fields, labels, linest = [], [], []
    fields += [['lkf_sensor_0', 'rkf_sensor_0', 'ekf_f_0',        'ekf_f_2',         'tsid_lkf_0',     'tsid_rkf_0']]
    labels += [['left force',   'right force',  'ekf left force', 'ekf right force', 'left des force', 'right des force']]
    linest += [['b', 'r', None, None, ':', ':']]
    fields += [['lkf_sensor_1', 'rkf_sensor_1', 'ekf_f_1',        'ekf_f_3',         'tsid_lkf_1',     'tsid_rkf_1']]
    labels += [['left force',   'right force',  'ekf left force', 'ekf right force', 'left des force', 'right des force']]
    linest += [['b', 'r', None, None, ':', ':']]
    plot_from_logger(lgr, dt, fields, labels, ['Forces Y', 'Forces Z'], linest, ncols=1)


if PLOT_COM_DERIVATIVES :
    ax_lbl = {0:'Y', 1:'Z'}    
    for i in [0,1]:
        fields, labels, linest = [], [], []
        fields += [['com_p_'+str(i),  'tsid_com_p_est_'+str(i)]]
        labels += [['com '+ax_lbl[i], 'estimated com '+ax_lbl[i]]]
        linest += [[None, '--']]
        fields += [['com_v_'+str(i),      'tsid_com_v_est_'+str(i)]]
        labels += [['com vel '+ax_lbl[i], 'estimated com vel '+ax_lbl[i]]]
        linest += [[None, '--']]
        fields += [['com_a_'+str(i),      'tsid_com_a_est_'+str(i)]]
        labels += [['com acc '+ax_lbl[i], 'estimated com acc '+ax_lbl[i]]]
        linest += [[None, '--']]
        fields += [['com_j_'+str(i),      'tsid_com_j_est_'+str(i)]]
        labels += [['com jerk '+ax_lbl[i], 'estimated com jerk '+ax_lbl[i]]]
        linest += [[None, '--']]
        plot_from_logger(lgr, dt, fields, labels, ['CoM', 'CoM Vel', 'CoM Acc', 'CoM Jerk'], linest, ncols=1)
        

if PLOT_ANGULAR_MOMENTUM_DERIVATIVES:
    f, ax = plt.subplots(5,1,sharex=True);
    ax1 = plt.subplot(511)
    plt.plot(log_t,lgr.tsid_iam, label="iam ") 
    plt.legend()
    plt.subplot(512,sharex=ax1)
    plt.plot(log_t,lgr.tsid_am, label="am ") 
    plt.plot(log_t,finite_diff(lgr.tsid_iam, dt),':', label="fd iam ") 
    plt.legend()
    plt.subplot(513,sharex=ax1)
    plt.plot(log_t,lgr.tsid_dam, label="dam ") 
    plt.plot(log_t,finite_diff(lgr_tsid_am, dt),':', label="fd am " ) 
    plt.legend()
    plt.subplot(514,sharex=ax1)
    plt.plot(log_t,lgr.tsid_ddam, label="ddam ") 
    plt.plot(log_t,finite_diff(lgr.tsid_dam,dt),':', label="fd dam ") 
    plt.legend()
    plt.subplot(515,sharex=ax1)
    plt.plot(log_t,lgr.tsid_dddam_des, label="desired dddam")
    plt.plot(log_t,finite_diff(lgr.tsid_ddam,dt),':', label="fd ddam")
    plt.legend()

if(PLOT_JOINT_TORQUES):
    plot_from_logger(lgr, dt, [['tsid_tau_'+str(i) for i in range(4)]])
    
plt.show()

try:
    embed()
except:
    pass;
