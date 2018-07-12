import pinocchio as se3
from numpy import matlib
import numpy as np
from numpy.linalg import norm
from pinocchio.utils import *
from math import pi,sqrt
from hrp2_reduced import Hrp2Reduced
import time 
from simu import Simu
from utils.utils_thomas import restert_viewer_server, traj_sinusoid, traj_norm, finite_diff
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
ZETA = .2   # with zeta=0.03 and ndt=100 it is unstable
tauc = 0*3.0*np.array([1.,1.,1.,1.])#coulomb friction

PLOT_FORCES                         = 1
PLOT_COM_ESTIMATION                 = 1
PLOT_COM_TRACKING                   = 1
PLOT_CONTACT_POINT_ACC              = 1
PLOT_ANGULAR_MOMENTUM_DERIVATIVES   = 1
PLOT_JOINT_TORQUES                  = 1
plut.SAVE_FIGURES                   = 0
SAVE_DATA                           = 0
SHOW_FIGURES                        = 1

#Simulation parameters
dt  = 1e-3
ndt = 30
simulation_time = 2.0
USE_ESTIMATOR = 1              # use real state for controller feedback
T_DISTURB_BEGIN = 0.20          # Time at which the disturbance starts
T_DISTURB_END   = 0.21          # Time at which the disturbance ends

INPUT_PARAMS = ['controller=', 'com_sin_amp=', 'f_dist=', 'zeta=', 'use_estimator=', 'T=']
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
    elif opt == "--T":
        simulation_time = float(arg);

print "*** CURRENT CONFIGURATION ***"
print "- controller =   ", CONTROLLER
print "- com_sin_amp =  ", COM_SIN_AMP[0]
print "- f_dist =       ", F_DISTURB[0,0]
print "- zeta =         ", ZETA
print "- use estimator =", USE_ESTIMATOR
print "- T =            ", simulation_time
print "- tau_c =        ", tauc
print "\n"
     
TEST_DESCR_STR = CONTROLLER + '_zeta_'+str(ZETA)
if(COM_SIN_AMP[0]!=0.0):
    TEST_DESCR_STR += '_comSinAmp_'+str(COM_SIN_AMP[0])
if(F_DISTURB[0,0]!=0.0):
    TEST_DESCR_STR += '_fDist_'+str(F_DISTURB[0,0])
    
if(SAVE_DATA):
    date_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S');
#    RESULTS_PATH = os.getcwd()+'/data/'+date_time+'_'+TEST_DESCR_STR+'/'
    RESULTS_PATH = os.getcwd()+'/data/'+TEST_DESCR_STR+'/'
    print "Gonna save results in folder:", RESULTS_PATH
    os.makedirs(RESULTS_PATH);
    plut.FIGURE_PATH = RESULTS_PATH

#TEST_DESCR_STR = '_'+TEST_DESCR_STR
TEST_DESCR_STR = ''
    
robot = Hrp2Reduced(urdf,[pkg],loadModel=True,useViewer=useViewer)
robot.display(robot.q0)
if useViewer:
    robot.viewer.setCameraTransform(0,[1.9154722690582275, -0.2266872227191925, 0.1087859719991684,
                                       0.5243823528289795, 0.518651008605957, 0.4620114266872406, 0.4925136864185333])

#robot parameters
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
sigma_ddf   = 1e2*np.ones(4)          # control (i.e. force accelerations) noise std dev used in EKF
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
Kp_com = 50.0                  # com proportional feedback gain
Kd_com = 2*sqrt(Kp_com)         # com derivative feedback gain
if(CONTROLLER=='tsid_flex'):
    w_post  = 0.3
#    (Kp_com, Kd_com, Ka_com, Kj_com) = (10611.05989124,  4182.20596787,   618.10999684,    40.5999999)     # poles [-10.3 -10.2 -10.1  -10.]
#    (Kp_com, Kd_com, Ka_com, Kj_com) = (52674.83686644, 13908.30537877,  1377.10995895, 60.5999991) # poles [-15.30235117 -15.19247204 -15.10739361 -14.99778229]
#    (Kp_com, Kd_com, Ka_com, Kj_com) = (1.64844157e+05, 3.27244115e+04, 2.43611027e+03, 8.06000045e+01) # ploes [-20.29999909 -20.20000804 -20.09999556 -20.00000185]
    (Kp_com, Kd_com, Ka_com, Kj_com) = (63057.32417634, 21171.16628651,  2076.59405021,    77.91842474) # poles 5, 15, 25, 35
#    (Kp_com, Kd_com, Ka_com, Kj_com) = (2.28323600e+05, 4.76834812e+04, 3.35391925e+03, 9.68316039e+01) # poles 10, 20, 30, 40
#    (Kp_com, Kd_com, Ka_com, Kj_com) = (7.78064620e+05, 1.03624061e+05, 5.18844741e+03, 1.16188573e+02) # poles -30
#    (Kp_com, Kd_com, Ka_com, Kj_com) = (1.20e+06, 1.54e+05, 7.10e+03, 1.40e+02) # poles [-50. -40. -30. -20.]
    tsid = TsidFlexibleContact(robot, Ky, Kz, w_post, Kp_post, Kp_com, Kd_com, Ka_com, Kj_com, centroidalEstimator)
elif(CONTROLLER=='tsid_rigid'):
    w_post  = 1e-2                  # postural task weight
    w_force = 1e-4
    tsid = Tsid(robot, Ky, Kz, w_post, w_force, Kp_post, Kp_com, centroidalEstimator)
elif(CONTROLLER=='tsid_adm'):
    Kf = np.matrix(np.diagflat([1.0/Ky, 1.0/Kz, 1.0/Ky, 1.0/Kz]))   # Stiffness of the feet spring
#    Kp_adm, Kd_adm, Kp_com, Kd_com = 1676.95962612, 96.8316038567, 136.153307873, 28.4344837195 # poles 10, 20, 30, 40
#    Kp_adm, Kd_adm, Kp_com, Kd_com = 1038.29702511, 77.9184247381, 60.731488824, 20.3902792501  # poles 5, 15, 25, 35
#    Kp_adm, Kd_adm, Kp_com, Kd_com, Kf = 188.781277292, 77.9184247381, 33.4023188532, 11.2146535876, 10*Kf # poles 5, 15, 25, 35
    Kp_adm, Kd_adm, Kp_com, Kd_com, Kf = 20.5603371308, 77.9184247381, 30.6694018561, 10.2970910213, 100*Kf # poles 5, 15, 25, 35
    tsid = TsidAdmittance(robot, Ky, Kz, w_post, Kp_post, Kp_com, Kf, Kp_adm, Kd_adm, centroidalEstimator)
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
tsid_var_names  = ['dv', 'tau', 'com_p_mes', 'com_v_mes', 'com_p_est', 'com_v_est', 'comref']
tsid_var_types  = 7*[vc,]

if CONTROLLER=='tsid_rigid' or CONTROLLER=='tsid_adm':
    tsid_var_names  += ['lkf', 'rkf']
    tsid_var_types  += 2*[vc,]
if CONTROLLER=='tsid_mistry':
    tsid_var_names += ['com_j_des', 'com_j_exp', 'df_des']
    tsid_var_types += [     vc    ,   vc       ,    vc]
if CONTROLLER=='tsid_flex' or CONTROLLER=='tsid_mistry':
    tsid_var_names += [ 'lf_a_des', 'rf_a_des']
    tsid_var_types += [     vc,         vc    ]
if CONTROLLER=='tsid_flex':
    #Integral of angular momentum approximated by base orientation, angular momentum, its 1st and 2nd derivative, its desired 3rd derivative
    tsid_var_names += ['iam', 'dddam_des', 'robotInertia']
    tsid_var_types += 3*[vr,]
    tsid_var_names += ['com_a_mes', 'com_a_est', 'com_j_est', 'com_s_des', 'com_s_exp']
    tsid_var_types += 5*[vc,]
else:
    tsid_var_names += ['com_a_des']
    tsid_var_types += [vc]
lgr.auto_log_variables(tsid.data, tsid_var_names, tsid_var_types, 'tsid')

lgr.auto_log_variables(simu, ['dv', 'com_p', 'com_v', 'com_a', 'com_j', 'com_s', 'vlf', 'vrf', 'acc_lf', 'acc_rf', 'df'], 11*[vc,], 'simu')
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


# compute CoM tracking error
com_err = np.empty((log_size,2))
com_err[:,0] = np.array(lgr.simu_com_p_0) - np.array(lgr.tsid_comref_0)
com_err[:,1] = np.array(lgr.simu_com_p_1) - np.array(lgr.tsid_comref_1)
com_err_norm = traj_norm(com_err)
print "CoM tracking RMSE:    %.1f mm"%(1e3*np.mean(com_err_norm))
print "Max CoM tracking err: %.1f mm"%(1e3*np.max(com_err_norm))

if PLOT_FORCES:
    fields, labels, linest = [], [], []
    fields += [['simu_lkf_0',          'tsid_lkf_0',     'simu_rkf_0',          'tsid_rkf_0']]
    labels += [['left',                'left des',       'right',               'right des']]
    linest += [['b', '--', 'r', '--']]
    fields += [['simu_lkf_1',          'tsid_lkf_1',     'simu_rkf_1',          'tsid_rkf_1']]
    labels += [['left',                'left des',       'right',               'right des']]
    linest += [['b', '--', 'r', '--']]
    plot_from_logger(lgr, dt, fields, labels, 'Contact Forces', linest, ylabel=['Y [N]', 'Z [N]'])
    plut.saveFigure('contact_forces'+TEST_DESCR_STR)

    fields, labels, linest = [], [], []
    fields += [['simu_lkf_0',   'ekf_f_0',    'simu_rkf_0',   'ekf_f_2'  ]]
    labels += [['left',         'ekf left',   'right',        'ekf right']]
    linest += [['b', '--', 'r', '--']]
    fields += [['simu_lkf_1',   'ekf_f_1',    'simu_rkf_1',   'ekf_f_3'  ]]
    labels += [['left',         'ekf left',   'right',        'ekf right']]
    linest += [['b', '--', 'r', '--']]
    plot_from_logger(lgr, dt, fields, labels, 'Contact Forces', linest, ylabel=['Y [N]', 'Z [N]'])
    plut.saveFigure('contact_forces_est'+TEST_DESCR_STR)


if PLOT_CONTACT_POINT_ACC:
    ax_lbl = {0:'Y', 1:'Z'}
    f_names = ['simu_acc_lf_', 'simu_acc_rf_', 'tsid_lf_a_des_', 'tsid_rf_a_des_']
    fields  = [[s+'0' for s in f_names]] + [[s+'1' for s in f_names]]
    labels = 2*[['real left', 'real right', 'des left', 'des right']]
    linest = 2*[['b',         'r',          '--',       '--'       ]]
    plot_from_logger(lgr, dt, fields, labels, 'Contact Point Accelerations', linest, ylabel=[s+' [m/s^2]' for s in ['Y','Z']])
    plut.saveFigure('contact_point_acc'+TEST_DESCR_STR)

if PLOT_COM_TRACKING :
    ax_lbl = {0:'Y', 1:'Z'}    
    for i in [0,1]:
        fields, labels, linest = [], [], []
        fields += [['simu_com_p_'+str(i),  'tsid_comref_'+str(i)]]
        labels += [['real',                'reference'          ]]
        linest += [[None,                  '--'                 ]]
        fields += [['simu_com_v_'+str(i) ]] + [['simu_com_a_'+str(i), 'tsid_com_a_des_'+str(i)]] + [['simu_com_j_'+str(i) ]]
        labels += [['real'               ]] + [['real',               'desired'               ]] + [['real'               ]]
        linest += [[None                 ]] + [[None,                 '--'                    ]] + [[None                 ]]
        ylabels = [s+' '+um for (s,um) in zip(['Pos.', 'Vel.', 'Acc.', 'Jerk'], ['[m]', '[m/s]', '[m/s^2]', '[m/s^3]'])]
        plot_from_logger(lgr, dt, fields, labels, titles='Center of Mass '+ax_lbl[i], linestyles=linest, ylabel=ylabels)
        plut.saveFigure('com_tracking_'+ax_lbl[i]+TEST_DESCR_STR)

if PLOT_COM_ESTIMATION and USE_ESTIMATOR:
    ax_lbl = {0:'Y', 1:'Z'}    
    for i in [0,1]:
        fields  = [['simu_com_p_'+str(i),  'tsid_com_p_est_'+str(i) ]] + [['simu_com_v_'+str(i),  'tsid_com_v_est_'+str(i) ]]
        fields += [['simu_com_a_'+str(i),  'tsid_com_a_est_'+str(i) ]] + [['simu_com_j_'+str(i),  'tsid_com_j_est_'+str(i) ]]
        labels = 4*[['real',      'estimated' ]]
        linest = 4*[[None, '--']]
        ylabels = [s+' '+um for (s,um) in zip(['Pos.', 'Vel.', 'Acc.', 'Jerk'], ['[m]', '[m/s]', '[m/s^2]', '[m/s^3]'])]
        plot_from_logger(lgr, dt, fields, labels, titles='Center of Mass '+ax_lbl[i], linestyles=linest, ylabel=ylabels)
        plut.saveFigure('com_estimate_'+ax_lbl[i]+TEST_DESCR_STR)

#plot_from_logger(lgr, dt, [['tsid_dv_'+str(i), 'simu_dv_'+str(i)] for i in range(4)])  
    
if CONTROLLER=='tsid_mistry':
    plot_from_logger(lgr, dt, [['tsid_com_j_des_'+str(i), 'tsid_com_j_exp_'+str(i), 'simu_com_j_'+str(i)] for i in range(1)], linestyles=[[None,'--',':']]*2)    
    plot_from_logger(lgr, dt, [['tsid_df_des_'+str(i), 'simu_df_'+str(i)] for i in range(4)], linestyles=[[None,'--']]*4, ncols=2)
elif CONTROLLER=='tsid_flex':
    plot_from_logger(lgr, dt, [['tsid_com_s_des_'+str(i), 'tsid_com_s_exp_'+str(i), 'simu_com_s_'+str(i)] for i in range(2)], linestyles=[[None,'--',':']]*2)
    
if PLOT_ANGULAR_MOMENTUM_DERIVATIVES:
    plot_from_logger(lgr, dt, [['simu_am'], ['simu_dam'], ['simu_ddam']])
    plut.saveFigure('angular_momentum'+TEST_DESCR_STR)

if(PLOT_JOINT_TORQUES):
    plot_from_logger(lgr, dt, [['tsid_tau_'+str(i) for i in range(4)]], [['right hip', 'right knee', 'left hip', 'left knee']], ylabel='Joint Generalized Force [N]/[Nm]')
    plut.saveFigure('joint_torques'+TEST_DESCR_STR)
    
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
