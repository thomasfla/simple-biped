import pinocchio as se3
from numpy import matlib
from pinocchio.utils import *
from math import pi,sqrt
from hrp2_reduced import Hrp2Reduced
import time 
from simu import Simu
from utils.utils_thomas import restert_viewer_server, traj_sinusoid, finite_diff
from utils.logger import RaiLogger
from utils.filters import FIR1LowPass, FiniteDiff
import matplotlib.pyplot as plt
import utils.plot_utils
from tsid import Tsid
from tsid_flexible_contacts import TsidFlexibleContact
from path import pkg, urdf 
from utils.noise_utils import NoisyState
from estimation.momentumEKF import *

try:
    from IPython import embed
except ImportError:
    pass

useViewer = False
np.set_printoptions(precision=3, linewidth=200)

if useViewer:
    restert_viewer_server()
    
def controller(q,v,f,df):
    ''' Take the sensor data (position, velocity and contact forces) 
    and generate a torque command '''
    t=log_index*simu.dt
    tsid.solve(q,v,f,df,t)
    
    if not log_index%100 :
        print "t:{0} \t com error \t{1} ".format(log_index*dt, np.linalg.norm(tsid.data.com_p_err))
        
    # check that the state doesn't go crazy
    if np.linalg.norm(tsid.data.com_p_err) > 0.1:
        raise ValueError("COM error > 0.1")
    if np.linalg.norm(tsid.data.f) > 500:
        raise ValueError("Forces > 500")
    if np.linalg.norm(q) > 10:
        raise ValueError("q > 10")
        
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

#Plots
PLOT_COM_AND_FORCES = 1
PLOT_COM_DERIVATIVES = 1
PLOT_ANGULAR_MOMENTUM_DERIVATIVES = 0   
   
#Simulation parameters
dt  = 1e-3
ndt = 5
simulation_time = 2.0
USE_REAL_STATE = 0       # use real state for controller feedback
T_DISTURB_BEGIN = 0.5           # Time at which the disturbance starts
T_DISTURB_END   = 0.51          # Time at which the disturbance ends
F_DISTURB = np.matrix([200.,0.,0.]).T

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
fc      = np.inf                # cutoff frequency of the Force fiter
Ktau    = 2.0                   # torque proportional feedback gain
Kdtau   = 2.*sqrt(Ktau)*0.00    # Make it unstable ??
Kp_post = 10                    # postural task proportional feedback gain
if(FLEXIBLE_CONTROLLER):
    w_post  = 0.1
    (Kp_com, Kd_com, Ka_com, Kj_com) = (1.20e+06, 1.54e+05, 7.10e+03, 1.40e+02)
else:
    w_post = 0.001                  # postural task weight
    Kp_com = 30                    # com proportional feedback gain
    Kd_com = 2*sqrt(Kp_com)         # com derivative feedback gain
    
COM_REF_START = [0.00, 0.53]
COM_REF_END   = [0.00, 0.53]
COM_TRAJ_TIME = 1.0

#Simulator
simu = Simu(robot,dt=dt,ndt=ndt)
simu.tauc = tauc
simu.Krf, simu.Klf = Kspring, Kspring
simu.Brf, simu.Blf = Bspring, Bspring

# size of configuration vector (NQ), velocity vector (NV), number of bodies (NB)
NQ,NV,NB,RF,LF,RK,LK = simu.NQ,simu.NV,simu.NB,simu.RF,simu.LF,simu.RK,simu.LK

#initial state
q0 = robot.q0.copy()
v0 = zero(NV)
g_vec = robot.model.gravity.linear[1:].A1            # gravity acc vector
g = np.linalg.norm(g_vec)                            # gravity acceleration
se3.computeAllTerms(robot.model,robot.data,q0,v0)
m = robot.data.mass[0] 
q0[1]-=0.5*m*g/Kz
f0,df0 = simu.compute_f_df_from_q_v(q0,v0)
c0,dc0,ddc0,dddc0 = robot.get_com_and_derivatives(q0,v0,f0,df0)
l0 = 0

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
sigma_ddf = 1e4*ones(4)             # control (i.e. force accelerations) noise std dev used in EKF
sigma_c  = 1e-3*ones(2)             # CoM position measurement noise std dev
sigma_dc = ns.std_gyry*ones(2)      # CoM velocity measurement noise std dev
sigma_l  = 1e-1*ones(1)             # angular momentum measurement noise std dev
sigma_f  = np.array([ns.std_fy, ns.std_fz, ns.std_fy, ns.std_fz])  # force measurement noise std dev
S_0 = sigma_x_0**2 * np.eye(n_x)

centroidalEstimator = MomentumEKF(dt, m, g_vec, c0.A1, dc0.A1, np.array([l0]), f0.A1, S_0, sigma_c, sigma_dc, sigma_l, sigma_f, sigma_ddf)

log_size = int(simulation_time / dt)    #max simulation samples
log_t        = np.zeros([log_size,1])+np.nan  # time
log_index = 0  

if FLEXIBLE_CONTROLLER:
    tsid=TsidFlexibleContact(robot,Ky,Kz,w_post,Kp_post,Kp_com, Kd_com, Ka_com, Kj_com, centroidalEstimator)
else:
    tsid=Tsid(robot,Ky,Kz,Kp_post,Kp_com,w_post)

tsid.callback_com = lambda t : com_traj(t, COM_REF_START, COM_REF_END, COM_TRAJ_TIME)

# SETUP LOGGER
lgr = RaiLogger()
vc = 'vector'
vr = 'variable'
tsid_var_names  = ['dv', 'tau', 'com_p_err', 'com_v_err', 'com_p_mes', 'com_v_mes', 'com_p_est', 'com_v_est', 'comref']
tsid_var_types  = [ vc,     vc,     vc,          vc,          vc,            vc,           vc,         vc,       vc]
#Integral of angular momentum approximated by base orientation, angular momentum, its 1st and 2nd derivative, its desired 3rd derivative
tsid_var_names += ['iam', 'am', 'dam', 'ddam', 'dddam_des']
tsid_var_types += [  vr,   vr,    vr,     vr,       vr]
tsid_var_names += ['lkf', 'rkf', 'lf_a_des', 'rf_a_des', 'robotInertia']
tsid_var_types += [  vc,    vc,      vc,         vc,         vr]
if FLEXIBLE_CONTROLLER:
    tsid_var_names += ['com_a_err', 'com_j_err', 'com_a_mes', 'com_a_est', 'com_j_est', 'com_s_des']
    tsid_var_types += [     vc,          vc,          vc,          vc,          vc,            vc]
else:
    tsid_var_names += ['com_a_des']
    tsid_var_types += [     vc]
lgr.auto_log_variables(tsid.data, tsid_var_names, tsid_var_types, 'tsid')
lgr.auto_log_variables(simu, ['vlf', 'vrf', 'dv'], [vc, vc, vc], 'simu')
lgr.auto_log_variables(centroidalEstimator, ['x'], [vc], log_var_names=[['ekf_c_0', 'ekf_c_1', 'ekf_dc_0', 'ekf_dc_1', 'ekf_l', 'ekf_f_0', 'ekf_f_1',
                                                                         'ekf_f_2', 'ekf_f_3', 'ekf_df_0', 'ekf_df_1', 'ekf_df_2', 'ekf_df_3']])

lgr.auto_log_local_variables(['com_p', 'com_v', 'com_a', 'com_j', 'ddf_des', 'ddf'], [vc, vc, vc, vc, vc, vc])
lgr.auto_log_local_variables(['f'], [vc], log_var_names=[['lkf_sensor_0', 'lkf_sensor_1', 'rkf_sensor_0', 'rkf_sensor_1']])
lgr.auto_log_local_variables(['df'], [vc], log_var_names=[['lkdf_sensor_0', 'lkdf_sensor_1', 'rkdf_sensor_0', 'rkdf_sensor_1']])
    
def loop(q, v, f, df, niter, ndt=None, dt=None, tsleep=.9, fdisplay=100):
    global log_index
    last_df  = np.matrix(np.zeros(4)).T
    ddf_des = np.zeros(4)
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
        
        # simulate the system
        u = control(q_noisy,v_noisy,f_noisy,df_noisy)
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
control = controller
q,v,f,df = q0.copy(), v0.copy(), f0.copy(), df0.copy()
q,v = loop(q,v,f,df,log_size)


if PLOT_COM_AND_FORCES:
    f, ax = plt.subplots(3,1,sharex=True);
    ax1 = plt.subplot(311)
    infostr = "Infos:"
    infostr += "\n Ktau  = {}".format(Ktau)
    infostr += "\n fc FTfilter = {} Hz".format(fc)
    infostr += "\n Kp_post {}".format(Kp_post)
    infostr += "\n Kp_com {}".format(Kp_com)
    infostr += "\n tauc {}".format(tauc)
    infostr += "\n Ky={} Kz={}".format(Ky,Kz)
    infostr += "\n dt={}ms ndt={}".format(dt*1000,ndt)
    plt.text(0.1, 0.05, infostr,
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.title('com tracking Y')
    plt.plot(log_t, lgr.com_p_0, label="com")
#    plt.plot(log_t, lgr.tsid_com_p_err_0, label="com error")
    plt.plot(log_t, lgr.tsid_comref_0,    label="com ref")
    plt.legend()
    plt.subplot(312, sharex=ax1)
    plt.title('feet forces Y') 
    plt.plot(log_t, lgr.get_streams('lkf_sensor_0'),label="left force", color='b')
    plt.plot(log_t, lgr.get_streams('rkf_sensor_0'),label="right force", color='r')
    plt.plot(log_t, lgr.ekf_f_0, '--', label="ekf left force")
    plt.plot(log_t, lgr.ekf_f_2, '--', label="ekf right force")
    plt.plot(log_t, lgr.tsid_lkf_0, label="left des force", linestyle=':')
    plt.plot(log_t, lgr.tsid_rkf_0, label="right des force", linestyle=':')
    plt.legend()
    plt.subplot(313, sharex=ax1)
    plt.title('feet forces Z')
    plt.plot(log_t, lgr.get_streams('lkf_sensor_1'), label="left force", color='b')
    plt.plot(log_t, lgr.get_streams('rkf_sensor_1'), label="right force", color='r')
    plt.plot(log_t, lgr.ekf_f_1, '--', label="ekf left force")
    plt.plot(log_t, lgr.ekf_f_3, '--', label="ekf right force")
    plt.plot(log_t, lgr.tsid_lkf_1, label="left des force", linestyle=':')
    plt.plot(log_t, lgr.tsid_rkf_1, label="right des force", linestyle=':')
    plt.legend()
#    plt.show()


if PLOT_COM_DERIVATIVES :
    axe_label = {0:'Y', 1:'Z'}    
    for i in [0,1]:
        f, ax = plt.subplots(5,1,sharex=True);
        ax1 = plt.subplot(511)
        plt.plot(log_t, lgr.get_streams('com_p_'+str(i)), label="com " + axe_label[i]) 
        plt.plot(log_t, lgr.get_streams('tsid_com_p_est_'+str(i)), '--', label="estimated com "+axe_label[i])
#        plt.plot(log_t, lgr.get_streams('tsid_com_p_mes_'+str(i)), label="measured com "+axe_label[i])        
#        plt.plot(log_t, lgr.get_streams('ekf_c_'+str(i)),          label="ekf com "+axe_label[i])        
        plt.legend()
        plt.subplot(512,sharex=ax1)
        plt.plot(log_t, lgr.get_streams('com_v_'+str(i)), label="vcom "+ axe_label[i]) 
        plt.plot(log_t, lgr.get_streams('tsid_com_v_est_'+str(i)), '--', label="estimated vcom "+ axe_label[i]) 
#        plt.plot(log_t, lgr.get_streams('tsid_com_v_mes_'+str(i)), label="measured vcom "+ axe_label[i])         
#        plt.plot(log_t, lgr.get_streams('ekf_dc_'+str(i)),          label="ekf vcom "+axe_label[i])        
        plt.legend()
        plt.subplot(513,sharex=ax1)
        plt.plot(log_t, lgr.get_streams('com_a_'+str(i)), label="acom "+ axe_label[i]) 
        plt.plot(log_t, lgr.get_streams('tsid_com_a_est_'+str(i)), '--', label="estimated acom "+ axe_label[i]) 
#        plt.plot(log_t, lgr.get_streams('tsid_com_a_mes_'+str(i)), label="measured acom "+ axe_label[i])         
        if not FLEXIBLE_CONTROLLER :
            plt.plot(log_t, lgr.get_streams('tsid_com_a_des_'+str(i)), label="desired acom" + axe_label[i])
        plt.legend()
        plt.subplot(514,sharex=ax1)
        plt.plot(log_t, lgr.get_streams('com_j_'+str(i)), label="jcom "+ axe_label[i]) 
        plt.plot(log_t, lgr.get_streams('tsid_com_j_est_'+str(i)), '--', label="estimated jcom "+ axe_label[i]) 
        plt.legend()
        plt.subplot(515,sharex=ax1)
        if FLEXIBLE_CONTROLLER :
            plt.plot(log_t, lgr.get_streams('tsid_com_s_des_'+str(i)), label="desired scom" + axe_label[i])
        plt.plot(log_t, finite_diff(lgr.get_streams('com_j_'+str(i)), dt), ':', label="finite diff jcom " + axe_label[i])
        #~ plt.plot(log_t,log_lkf_sensor[:,1], label="force Left z")
        #~ plt.plot(log_t,log_rkf_sensor[:,1], label="force Right z")
        plt.legend()

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

plt.show()

try:
    embed()
except:
    pass;
