import pinocchio as se3
from numpy import matlib
import numpy as np
from numpy.linalg import norm
from pinocchio.utils import *
from math import pi,sqrt
from hrp2_reduced import Hrp2Reduced
import time 
from simu import Simu
from utils.utils_thomas import restert_viewer_server, traj_sinusoid, finite_diff
from utils.logger import RaiLogger
import matplotlib.pyplot as plt
import utils.plot_utils as plut
from utils.plot_utils import plot_from_logger
from tsid import Tsid
from tsid_admittance import TsidAdmittance
from tsid_flexible_contacts import TsidFlexibleContact
from tsid_mistry import TsidMistry
from path import pkg, urdf 
from utils.noise_utils import NoisyState
from estimation.momentumEKF import MomentumEKF
import getopt, sys, os, datetime

try:
    from IPython import embed
except ImportError:
    pass

useViewer = False
np.set_printoptions(precision=3, linewidth=200)

if useViewer:
    restert_viewer_server()
    
def controller(t, q, v, f, df):
    ''' Take the sensor data (position, velocity and contact forces) 
    and generate a torque command '''
    tsid.solve(t, q, v, f, df)
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
    
CONTROLLER = 'tsid_flex'             # either 'tsid_rigid' or 'tsid_flex' or 'tsid_adm' or 'tsid_mistry'
F_DISTURB = np.matrix([0e2, 0, 0]).T
COM_SIN_AMP = np.array([0.03, 0.0])
ZETA = .3   # with zeta=0.03 and ndt=100 it is unstable

PLOT_FORCES                         = 1
PLOT_COM_ESTIMATION                 = 0
PLOT_COM_TRACKING                   = 1
PLOT_CONTACT_POINT_ACC              = 1
PLOT_ANGULAR_MOMENTUM_DERIVATIVES   = 0
PLOT_JOINT_TORQUES                  = 1
plut.SAVE_FIGURES                   = 1
SAVE_DATA                           = 1
SHOW_FIGURES                        = 0

#Simulation parameters
dt  = 1e-3
ndt = 10
simulation_time = 2.0
USE_ESTIMATOR = 0              # use real state for controller feedback
T_DISTURB_BEGIN = 0.20          # Time at which the disturbance starts
T_DISTURB_END   = 0.21          # Time at which the disturbance ends

INPUT_PARAMS = ['controller=', 'com_sin_amp=', 'f_dist=', 'zeta=', 'use_estimator=']
try:
    opts, args = getopt.getopt(sys.argv[1:],"",INPUT_PARAMS)
except getopt.GetoptError:
    print "Error while parsing command-line arguments."
    print 'Example of usage:'
    print '    track_com.py --controller tsid_flex';
    print 'These are the available input parameters:\n', INPUT_PARAMS;
    sys.exit(2);
    
for opt, arg in opts:
    if opt == '--controller':
        CONTROLLER = str(arg);
    elif opt == "--com_sin_amp":
        COM_SIN_AMP[0] = float(arg);
    elif opt == "--f_dist":
        F_DISTURB[0,0] = float(arg);
    elif opt == "--zeta":
        ZETA = float(arg);
    elif opt == "--use_estimator":
        USE_ESTIMATOR = bool(arg);

print "*** CURRENT CONFIGURATION ***"
print "- controller = ", CONTROLLER
print "- com_sin_amp =", COM_SIN_AMP[0]
print "- f_dist =     ", F_DISTURB[0,0]
print "- zeta =       ", ZETA
print "\n"
     
TEST_DESCR_STR = CONTROLLER + '_zeta_'+str(ZETA)
if(COM_SIN_AMP[0]!=0.0):
    TEST_DESCR_STR += '_comSinAmp_'+str(COM_SIN_AMP[0])
if(F_DISTURB[0,0]!=0.0):
    TEST_DESCR_STR += '_fDist_'+str(F_DISTURB[0,0])
    
if(SAVE_DATA):
    date_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S');
    RESULTS_PATH = os.getcwd()+'/data/'+date_time+'_'+TEST_DESCR_STR+'/'
    print "Gonna save results in folder:", RESULTS_PATH
    os.makedirs(RESULTS_PATH);
    plut.FIGURE_PATH = RESULTS_PATH
    
robot = Hrp2Reduced(urdf,[pkg],loadModel=True,useViewer=useViewer)
robot.display(robot.q0)
if useViewer:
    robot.viewer.setCameraTransform(0,[1.9154722690582275, -0.2266872227191925, 0.1087859719991684,
                                       0.5243823528289795, 0.518651008605957, 0.4620114266872406, 0.4925136864185333])



#robot parameters
tauc = 0.*np.array([1.,1.,1.,1.])#coulomb friction
Ky = 23770.
Kz = 239018.
By = ZETA*2*sqrt(Ky) #50e0
Bz = ZETA*2*sqrt(Kz) #500e0
Kspring = -np.diagflat([Ky,Kz,0.])     # Stiffness of the feet spring
Bspring = -np.diagflat([By,Bz,0.])     # damping of the feet spring

#Simulator
simu = Simu(robot,dt=dt,ndt=ndt)
simu.tauc = tauc
simu.Krf, simu.Klf = Kspring, Kspring
simu.Brf, simu.Blf = Bspring, Bspring

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

COM_REF_START = c0.A1
COM_REF_END   = c0.A1 + COM_SIN_AMP
COM_TRAJ_TIME = 1.0

#Noise applied on the state to get a simulated measurement
ns = NoisyState(dt,robot,Ky,Kz)
# noise standard deviation
n_x, n_u, n_y = 9+4, 4, 9
sigma_x_0 = 1e-2                    # initial state estimate std dev
sigma_ddf   = 1e4*np.ones(4)          # control (i.e. force accelerations) noise std dev used in EKF
sigma_f     = np.array([ns.std_fy, ns.std_fz, ns.std_fy, ns.std_fz])  # force measurement noise std dev
sigma_f_dist = 1e1*np.ones(2)          # external force noise std dev used in EKF
sigma_c  = 1e-3*np.ones(2)             # CoM position measurement noise std dev
sigma_dc = ns.std_gyry*np.ones(2)      # CoM velocity measurement noise std dev
sigma_l  = 1e-1*np.ones(1)             # angular momentum measurement noise std dev
S_0 = sigma_x_0**2 * np.eye(n_x)

centroidalEstimator = MomentumEKF(dt, m, g_vec, c0.A1, dc0.A1, np.array([l0]), f0.A1, S_0, sigma_c, sigma_dc, sigma_l, sigma_f, sigma_ddf, sigma_f_dist)

#Controller parameters
w_post = 0.001                  # postural task weight
Kp_post = 10                    # postural task proportional feedback gain
Kp_com = 30                    # com proportional feedback gain
Kd_com = 2*sqrt(Kp_com)         # com derivative feedback gain
if(CONTROLLER=='tsid_flex'):
    w_post  = 0.3
    (Kp_com, Kd_com, Ka_com, Kj_com) = (1.20e+06, 1.54e+05, 7.10e+03, 1.40e+02)
    tsid = TsidFlexibleContact(robot, Ky, Kz, w_post, Kp_post, Kp_com, Kd_com, Ka_com, Kj_com, centroidalEstimator)
elif(CONTROLLER=='tsid_rigid'):
    w_post  = 1e-2                  # postural task weight
    w_force = 1e-4
    tsid = Tsid(robot, Ky, Kz, w_post, w_force, Kp_post, Kp_com, centroidalEstimator)
elif(CONTROLLER=='tsid_adm'):
    Kp_adm = 1000.0                 # with kp_amd=1500 it starts being unstable
    Kd_adm = 2*sqrt(Kp_adm)
    tsid = TsidAdmittance(robot, Ky, Kz, w_post, Kp_post, Kp_com, Kp_adm, Kd_adm, centroidalEstimator)
elif(CONTROLLER=='tsid_mistry'):
    w_post  = 1e-3                  # postural task weight
    tsid = TsidMistry(robot, Ky, Kz, By, Bz, w_post, Kp_post, Kp_com, Kd_com, dt, centroidalEstimator)

if(not USE_ESTIMATOR):
    tsid.estimator = None
    
log_size = int(simulation_time / dt)    #max simulation samples
log_t        = np.zeros([log_size,1])+np.nan  # time
tsid.callback_com = lambda t : com_traj(t, COM_REF_START, COM_REF_END, COM_TRAJ_TIME)

# SETUP LOGGER
lgr = RaiLogger()
vc, vr = 'vector', 'variable'
tsid_var_names  = ['dv', 'tau', 'com_p_mes', 'com_v_mes', 'com_p_est', 'com_v_est', 'comref', 'lkf', 'rkf']
tsid_var_types  = 9*[vc,]

if CONTROLLER=='tsid_mistry':
    tsid_var_names += ['com_j_des', 'com_j_exp', 'df_des']
    tsid_var_types += [     vc    ,   vc       ,    vc]
if CONTROLLER=='tsid_flex' or CONTROLLER=='tsid_mistry':
    tsid_var_names += [ 'lf_a_des', 'rf_a_des']
    tsid_var_types += [     vc,         vc    ]

if CONTROLLER=='tsid_flex':
    #Integral of angular momentum approximated by base orientation, angular momentum, its 1st and 2nd derivative, its desired 3rd derivative
    tsid_var_names += ['iam', 'dddam_des', 'robotInertia']
    tsid_var_types += [  vr,        vr,          vr]
    tsid_var_names += ['com_a_mes', 'com_a_est', 'com_j_est', 'com_s_des']
    tsid_var_types += [       vc,          vc,          vc,            vc]
else:
    tsid_var_names += ['com_a_des']
    tsid_var_types += [     vc]
lgr.auto_log_variables(tsid.data, tsid_var_names, tsid_var_types, 'tsid')

lgr.auto_log_variables(simu, ['dv', 'com_p', 'com_v', 'com_a', 'com_j', 'vlf', 'vrf', 'acc_lf', 'acc_rf', 'df'], 10*[vc,], 'simu')
lgr.auto_log_variables(simu, ['am', 'dam', 'ddam'], 3*[vr,], 'simu')
lgr.auto_log_variables(simu, ['f'], [vc], log_var_names=[['simu_'+s for s in ['lkf_0', 'lkf_1', 'rkf_0', 'rkf_1']]])

if(USE_ESTIMATOR):
    lgr.auto_log_variables(centroidalEstimator, ['x'], [vc], log_var_names=[['ekf_c_0', 'ekf_c_1', 'ekf_dc_0', 'ekf_dc_1', 'ekf_l', 'ekf_f_0', 'ekf_f_1',
                                                                             'ekf_f_2', 'ekf_f_3', 'ekf_df_0', 'ekf_df_1', 'ekf_df_2', 'ekf_df_3']])
    
def loop(q, v, f, df, niter, ndt=None, dt=None, tsleep=.9, fdisplay=100):
    t0 = time.time()
    if dt  is not None: simu.dt  = dt
    if ndt is not None: simu.ndt = ndt
    robot.display(q)
    
    for i in range(niter):
        t = i*simu.dt
        
        # add noise to the perfect state q,v,f,df
        if USE_ESTIMATOR:
            q_noisy,v_noisy,f_noisy,df_noisy = ns.get_noisy_state(q,v,f,df)
        else:
            q_noisy,v_noisy,f_noisy,df_noisy = q,v,f,df            
        
        # simulate the system
        u = controller(t, q_noisy, v_noisy, f_noisy, df_noisy)
        q,v,f,df = simu(q,v,u)
        
        # log data        
        log_t[i] = t  
        lgr.log_all(locals())
        
        if not i%100 :
            print "t:%.1f \t com err %.3f\t ang-mom %.1f\t tau norm %.0f" % (t, norm(tsid.data.com_p_err), simu.am, norm(tsid.data.tau))

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
    fields += [['simu_lkf_0',   'ekf_f_0',        'tsid_lkf_0']]
    labels += [['left force',   'ekf left force', 'left des force']]
    linest += [['b', '--', ':']]
    fields += [['simu_rkf_0',   'ekf_f_2',         'tsid_rkf_0']]
    labels += [['right force',  'ekf right force', 'right des force']]
    linest += [['b', '--', ':']]
    fields += [['simu_lkf_1',   'ekf_f_1',        'tsid_lkf_1',     ]]
    labels += [['left force',   'ekf left force', 'left des force'  ]]
    linest += [['r', '--', ':']]
    fields += [['simu_rkf_1',   'ekf_f_3',         'tsid_rkf_1']]
    labels += [['right force',  'ekf right force', 'right des force']]
    linest += [['r', '--', ':']]
    plot_from_logger(lgr, dt, fields, labels, ['Force Y Left', 'Force Y Right', 'Force Z Left', 'Force Z Right'], linest, ncols=2)
    plut.saveFigure('contact_forces_'+TEST_DESCR_STR)

if PLOT_CONTACT_POINT_ACC:
    ax_lbl = {0:'Y', 1:'Z'}    
    for i in [0,1]:
        fields = [['tsid_lf_a_des_'+str(i),         'simu_acc_lf_'+str(i)       ]] + [['tsid_rf_a_des_'+str(i),         'simu_acc_rf_'+str(i)       ]]
        labels = [['des left foot acc '+ax_lbl[i],  'left foot acc '+ax_lbl[i]  ]] + [['des right foot acc '+ax_lbl[i], 'right foot acc '+ax_lbl[i] ]]
        linest = [[None,                            '--'                        ]] + [[None,                            '--'                        ]]
        plot_from_logger(lgr, dt, fields, labels, ['Contact Point Accelerations', 'Contact Point Accelerations'], linest, ncols=1)
        plut.saveFigure('contact_point_acc_'+ax_lbl[i]+'_'+TEST_DESCR_STR)
        
if PLOT_COM_TRACKING :
    ax_lbl = {0:'Y', 1:'Z'}    
    for i in [0,1]:
        fields, labels, linest = [], [], []
        fields += [['simu_com_p_'+str(i),  'tsid_comref_'+str(i) ]]
        labels += [['com '+ax_lbl[i], 'ref com '+ax_lbl[i]  ]]
        linest += [[None,             '--']]
        fields += [['simu_com_v_'+str(i)      ]] + [['simu_com_a_'+str(i), 'tsid_com_a_des_'+str(i)]] + [['simu_com_j_'+str(i)  ]]
        labels += [['com vel '+ax_lbl[i]      ]] + [['com acc '+ax_lbl[i], 'com acc des '+ax_lbl[i]]] + [['com jerk '+ax_lbl[i] ]]
        linest += [[None                      ]] + [[None,                 '--'                    ]] + [[None                  ]]
        plot_from_logger(lgr, dt, fields, labels, ['CoM', 'CoM Vel', 'CoM Acc', 'CoM Jerk'], linest, ncols=1)
        plut.saveFigure('com_tracking_'+ax_lbl[i]+'_'+TEST_DESCR_STR)

if PLOT_COM_ESTIMATION :
    ax_lbl = {0:'Y', 1:'Z'}    
    for i in [0,1]:
        fields, labels, linest = [], [], []
        fields += [['simu_com_p_'+str(i),  'tsid_com_p_est_'+str(i)   ]] + [['simu_com_v_'+str(i),  'tsid_com_v_est_'+str(i)      ]]
        labels += [['com '+ax_lbl[i],      'estimated com '+ax_lbl[i] ]] + [['com vel '+ax_lbl[i],  'estimated com vel '+ax_lbl[i]]]
        linest += [[None, '--']] + [[None, '--']]
        fields += [['simu_com_a_'+str(i),  'tsid_com_a_est_'+str(i)      ]]
        labels += [['com acc '+ax_lbl[i],  'estimated com acc '+ax_lbl[i]]]
        linest += [[None, '--']]
        fields += [['simu_com_j_'+str(i),  'tsid_com_j_est_'+str(i)       ]]
        labels += [['com jerk '+ax_lbl[i], 'estimated com jerk '+ax_lbl[i]]]
        linest += [[None, '--']]
        plot_from_logger(lgr, dt, fields, labels, ['CoM', 'CoM Vel', 'CoM Acc', 'CoM Jerk'], linest, ncols=1)
        plut.saveFigure('com_estimate_'+ax_lbl[i]+'_'+TEST_DESCR_STR)

#plot_from_logger(lgr, dt, [['tsid_dv_'+str(i), 'simu_dv_'+str(i)] for i in range(4)])  
    
if CONTROLLER=='tsid_mistry':
    plot_from_logger(lgr, dt, [['tsid_com_j_des_'+str(i), 'tsid_com_j_exp_'+str(i), 'simu_com_j_'+str(i)] for i in range(1)], linestyles=[[None,'--',':']]*2)    
    plot_from_logger(lgr, dt, [['tsid_df_des_'+str(i), 'simu_df_'+str(i)] for i in range(4)], linestyles=[[None,'--']]*4, ncols=2)

if PLOT_ANGULAR_MOMENTUM_DERIVATIVES:
    plot_from_logger(lgr, dt, [['simu_am'], ['simu_dam'], ['simu_ddam']])
    plut.saveFigure('angular_momentum_'+TEST_DESCR_STR)

if(PLOT_JOINT_TORQUES):
    plot_from_logger(lgr, dt, [['tsid_tau_'+str(i) for i in range(4)]], [['Joint torque '+str(i) for i in range(4)]])
    plut.saveFigure('joint_torques_'+TEST_DESCR_STR)
    
if(SAVE_DATA):
    lgr.dump_compressed(RESULTS_PATH+'logger_data')
#tfile = open(plot_utils.FIGURE_PATH+conf.TEXT_FILE_NAME, "w")
#tfile.write(info);
#tfile.close();

if(SHOW_FIGURES):
    plt.show()

try:
    embed()
except:
    pass;
