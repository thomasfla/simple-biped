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

from tsid import Tsid, GainsTsid
from tsid_admittance import TsidAdmittance, GainsTsidAdm
from admittance_ctrl import AdmittanceControl, GainsAdmCtrl
from tsid_flexible_contacts import TsidFlexibleContact, GainsTsidFlexK
from tsid_mistry import TsidMistry

from robot_model_path import pkg, urdf
from utils.noise_utils import NoisyState
from estimation.momentumEKF import MomentumEKF
import getopt, sys, os, datetime
from math import log

import simple_biped.gain_tuning.conf_common as conf

try:
    from IPython import embed
except ImportError:
    pass

useViewer = 0
np.set_printoptions(precision=3, linewidth=200)

if useViewer:
    restert_viewer_server()

def controller(t, q, v, f, df):
    ''' Take the sensor data (position, velocity and contact forces)
    and generate a torque command '''
    tsid.solve(t, q, v, f, df)
    return np.vstack([f_disturb_traj(t), tsid.data.tau])

def f_disturb_traj(t):
    if (t>=T_DISTURB_BEGIN and t<T_DISTURB_END ):
        return F_DISTURB
    return matlib.zeros(3).T

def com_traj(t, c_init, c_final, T):
    py,vy,ay,jy,sy = traj_sinusoid(t, c_init[0], c_final[0], T)
    pz,vz,az,jz,sz = traj_sinusoid(t, c_init[1], c_final[1], T)
    return (np.matrix([[py ],[pz]]), np.matrix([[vy ],[vz]]),
            np.matrix([[ay ],[az]]), np.matrix([[jy ],[jz]]), np.matrix([[sy ],[sz]]))

CONTROLLER = 'adm_ctrl'             # either 'tsid_rigid' or 'tsid_flex_k' or 'tsid_adm' or 'tsid_mistry' or 'adm_ctrl'
F_DISTURB = np.matrix([0e3, 0, 0]).T
COM_SIN_AMP = np.array([0.0, 0.0])
ZETA = conf.zetas[0]   # with zeta=0.03 and ndt=100 it is unstable
k = conf.k
mu_simu = conf.mu    # contact force friction coefficient
mu_ctrl = conf.mu    # contact force friction coefficient
fMin = 10.0
tauc = conf.joint_coulomb_friction
JOINT_TORQUES_CUT_FREQUENCY = conf.JOINT_TORQUES_CUT_FREQUENCY

PLOT_FORCES                         = 1
PLOT_FORCE_VEL                      = 0
PLOT_FORCE_ACC                      = 0
PLOT_COM_ESTIMATION                 = 1
PLOT_COM_TRACKING                   = 1
PLOT_COM_SNAP                       = 0
PLOT_CONTACT_POINT_ACC              = 0
PLOT_ANGULAR_MOMENTUM_DERIVATIVES   = 0
PLOT_JOINT_TORQUES                  = 1
PLOT_JOINT_ANGLES                   = 0
plut.SAVE_FIGURES                   = 1
SAVE_DATA                           = 1
SHOW_FIGURES                        = 0

#Simulation parameters
dt  = conf.dt_simu
ndt = conf.ndt
simulation_time = conf.T_simu
USE_ESTIMATOR = conf.USE_ESTIMATOR  # use real state for controller feedback
T_DISTURB_BEGIN = 0.10              # Time at which the disturbance starts
T_DISTURB_END   = 0.101             # Time at which the disturbance ends
gain_file = None #'/home/student/repos/simple_biped/gain_tuning/../data/gains/gains_tsid_flex_k_w_d4x=1e-09.npy'
test_name = None

INPUT_PARAMS = ['controller=', 'com_sin_amp=', 'f_dist=', 'zeta=', 'use_estimator=', 'T=', 'k=', 'gain_file=', 'test_name=']
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
    elif opt == "--k":
        k = float(arg);
    elif opt == "--use_estimator":
        USE_ESTIMATOR = bool(arg);
    elif opt == "--T":
        simulation_time = float(arg);
    elif opt == "--gain_file":
        gain_file = str(arg)[1:-1];
    elif opt == "--test_name":
        test_name = str(arg)[1:-1];

print "*** CURRENT CONFIGURATION ***"
print "- controller =   ", CONTROLLER
print "- com_sin_amp =  ", COM_SIN_AMP[0]
print "- f_dist =       ", F_DISTURB[0,0]
print "- zeta =         ", ZETA
print "- k =            ", k
print "- use estimator =", USE_ESTIMATOR
print "- T =            ", simulation_time
print "- tau_c =        ", tauc
print "- gain_file =    ", gain_file
print "- test_name =    ", test_name
print "\n"

if(test_name is None):
    date_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S');
    test_name = date_time+'_'+CONTROLLER + '_zeta_'+str(ZETA) + '_k_'+str(k)
    if(COM_SIN_AMP[0]!=0.0):
        test_name += '_comSinAmp_'+str(COM_SIN_AMP[0])
    if(F_DISTURB[0,0]!=0.0):
        test_name += '_fDist_'+str(F_DISTURB[0,0])

if(SAVE_DATA):
    RESULTS_PATH = str(os.path.dirname(os.path.abspath(__file__)))+'/data/'+test_name+'/'
    print "Gonna save results in folder:", RESULTS_PATH
    try:
        os.makedirs(RESULTS_PATH);
    except OSError:
        print "Directory already exists so I am going to overwrite existing data."
#        raw_input("Are you sure that you want to continue? ")
    plut.FIGURE_PATH = RESULTS_PATH

robot = Hrp2Reduced(urdf,[pkg],loadModel=True,useViewer=useViewer)
robot.display(robot.q0)
if useViewer:
    robot.viewer.setCameraTransform(0,[1.9154722690582275, -0.2266872227191925, 0.1087859719991684,
                                       0.5243823528289795, 0.518651008605957, 0.4620114266872406, 0.4925136864185333])

if(k>1.0):
    ndt = int(ndt*(1+log(k)))
    print 'Increasing ndt to %d'%(ndt)

#robot parameters
Ky = conf.Ky
Kz = conf.Kz
By = ZETA*2*sqrt(Ky) #50e0
Bz = ZETA*2*sqrt(Kz) #500e0
K = np.asmatrix(np.diagflat([Ky,Kz,Ky,Kz]))
Kspring = -np.diagflat([Ky,Kz,0.])     # Stiffness of the feet spring
Bspring = -np.diagflat([By,Bz,0.])     # damping of the feet spring

#Simulator
simu = Simu(robot,dt=dt,ndt=ndt)
simu.tauc = tauc
simu.Krf, simu.Klf = Kspring, Kspring
simu.Brf, simu.Blf = Bspring, Bspring
simu.enable_friction_cones(mu_simu)
simu.joint_torques_cut_frequency = JOINT_TORQUES_CUT_FREQUENCY

#initial state
#q0 = robot.q0.copy()
#v0 = zero(robot.model.nv)
g_vec = robot.model.gravity.linear[1:].A1            # gravity acc vector
g = np.linalg.norm(g_vec)                            # gravity acceleration
#se3.computeAllTerms(robot.model,robot.data,q0,v0)
m = robot.data.mass[0]
q0 = conf.q0
v0 = conf.v0

# project joint velocities in null space of contact Jacobian
Jl,Jr = robot.get_Jl_Jr_world(q0)
Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
Nc = matlib.eye(robot.model.nv) - np.linalg.pinv(Jc)*Jc
v0 = Nc * v0

simu.init(q0, v0)
f0,df0 = simu.f, simu.df #compute_f_df_from_q_v(q0,v0)
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
gains_array = None
if(gain_file is not None):
    try:
        gains_array = np.load(gain_file)
    except:
        print "Error while trying to read gain file:", gain_file

w_post = 0.001                  # postural task weight
Kp_post = 10                    # postural task proportional feedback gain
if(CONTROLLER=='tsid_flex_k'):
    w_post  = 0.3
    ddf_max = 2e4 #4e5 #Kz
    if(gains_array is None):    gains = GainsTsidFlexK.get_default_gains()
    else:                       gains = GainsTsidFlexK(gains_array)
    tsid = TsidFlexibleContact(robot, Ky, Kz, w_post, Kp_post, gains, fMin, mu_ctrl, ddf_max, dt, centroidalEstimator)
elif(CONTROLLER=='tsid_rigid'):
    w_post  = 1e-2                  # postural task weight
    w_force = 1e-4
    if(gains_array is None):    gains = GainsTsid.get_default_gains()
    else:                       gains = GainsTsid(gains_array)
    tsid = Tsid(robot, Ky, Kz, w_post, w_force, Kp_post, gains, centroidalEstimator)
elif(CONTROLLER=='tsid_adm'):
    if(gains_array is None):    gains = GainsTsidAdm.get_default_gains(K)
    else:                       gains = GainsTsidAdm(gains_array)
    tsid = TsidAdmittance(robot, Ky, Kz, w_post, Kp_post, gains, dt, fMin, mu_ctrl, centroidalEstimator)
elif(CONTROLLER=='adm_ctrl'):
    if(gains_array is None):    gains = GainsAdmCtrl.get_default_gains(K)
    else:                       gains = GainsAdmCtrl(gains_array)

    Mj_diag = np.matrix(np.diag(np.diag(robot.data.M[3:,3:])))
    gains.Kp_pos = gains.kp_bar*Mj_diag
    gains.Kd_pos = gains.kd_bar*Mj_diag

    tau_0 = np.matrix([-2.590388574401987, -214.40477481684383, 2.570829716838424, -214.5015213169777]).T
    q_cmd = q0.copy()
    q_cmd[4:,0] += np.matrix(np.divide(tau_0.A1, np.diag(gains.Kp_pos))).T

    tsid = AdmittanceControl(robot, dt, q_cmd, Ky, Kz, w_post, Kp_post, gains, fMin, mu_ctrl, centroidalEstimator)
elif(CONTROLLER=='tsid_mistry'):
    Kp_com = 50.0                  # com proportional feedback gain
    Kd_com = 2*sqrt(Kp_com)         # com derivative feedback gain
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

if CONTROLLER=='adm_ctrl':
    tsid_var_names  += ['q_cmd']
    tsid_var_types  += 1*[vc,]
if CONTROLLER=='tsid_rigid' or CONTROLLER=='tsid_adm' or CONTROLLER=='adm_ctrl':
    tsid_var_names  += ['lkf', 'rkf']
    tsid_var_types  += 2*[vc,]
if CONTROLLER=='tsid_mistry':
    tsid_var_names += ['com_j_des', 'com_j_exp', 'df_des']
    tsid_var_types += [     vc    ,   vc       ,    vc]
if CONTROLLER=='tsid_flex_k' or CONTROLLER=='tsid_mistry':
    tsid_var_names += [ 'lf_a_des', 'rf_a_des']
    tsid_var_types += [     vc,         vc    ]
if CONTROLLER=='tsid_flex_k':
    #Integral of angular momentum approximated by base orientation, angular momentum, its 1st and 2nd derivative, its desired 3rd derivative
    tsid_var_names += ['iam', 'dddam_des', 'robotInertia']
    tsid_var_types += 3*[vr,]
    tsid_var_names += ['com_a_mes', 'com_a_est', 'com_j_est', 'com_s_des', 'com_s_exp', 'ddf_des']
    tsid_var_types += 6*[vc,]
    tsid_var_names += ['B_ddf_max', 'B_ddf_des', 'B_ddf_UB', 'B_df', 'B_f', 'B_df_max']
    tsid_var_types += 6*[vc,]
else:
    tsid_var_names += ['com_a_des']
    tsid_var_types += [vc]
lgr.auto_log_variables(tsid.data, tsid_var_names, tsid_var_types, 'tsid')

lgr.auto_log_variables(simu, ['q', 'v', 'dv', 'tauq', 'vlf', 'vrf', 'acc_lf', 'acc_rf', 'df', 'ddf'], 10*[vc,], 'simu')
lgr.auto_log_variables(simu, ['com_p', 'com_v', 'com_a', 'com_j', 'com_s'], 5*[vc,], 'simu')
lgr.auto_log_variables(simu, ['am', 'dam', 'ddam'], 3*[vr,], 'simu')
lgr.auto_log_variables(simu, ['f'], [vc], log_var_names=[['simu_'+s for s in ['lkf_0', 'lkf_1', 'rkf_0', 'rkf_1']]])

if(USE_ESTIMATOR):
    lgr.auto_log_variables(centroidalEstimator, ['x'], [vc], log_var_names=[['ekf_c_0', 'ekf_c_1', 'ekf_dc_0', 'ekf_dc_1', 'ekf_l', 'ekf_f_0', 'ekf_f_1',
                                                                             'ekf_f_2', 'ekf_f_3', 'ekf_df_0', 'ekf_df_1', 'ekf_df_2', 'ekf_df_3']])

def loop(q, v, f, df, niter, ndt=None, dt=None, fdisplay=10):
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

        # log data
        log_t[i] = t
        lgr.log_all(locals())

        q,v,f,df = simu(q,v,u)

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
    ax = plot_from_logger(lgr, dt, fields, labels, 'Contact Forces', linest, ylabel=['Y [N]', 'Z [N]'])
    plut.saveFigure('contact_forces')

    tt = np.arange(0.0, dt*log_size, dt)
    ax[0].plot(tt,  mu_simu*np.array(lgr.simu_lkf_1), 'b:', label='left bounds')
    ax[0].plot(tt, -mu_simu*np.array(lgr.simu_lkf_1), 'b:') #, label='left min')
    ax[0].plot(tt,  mu_simu*np.array(lgr.simu_rkf_1), 'r:', label='right bounds')
    ax[0].plot(tt, -mu_simu*np.array(lgr.simu_rkf_1), 'r:') #, label='right min')
#    ax[0].legend()    
    plut.saveFigure('contact_forces_with_friction_bounds')
    
    f, ax = plt.subplots(1, 1, sharex=True);
    ax.plot(tt, np.divide(lgr.simu_lkf_0, lgr.simu_lkf_1), label='left')
    ax.plot(tt, np.divide(lgr.simu_rkf_0, lgr.simu_rkf_1), label='right')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force Y / Z')
    ax.legend(loc='best')
    plut.saveFigure('contact_forces_ratio')

if(PLOT_FORCES and USE_ESTIMATOR):
    fields, labels, linest = [], [], []
    fields += [['simu_lkf_0',   'ekf_f_0',    'simu_rkf_0',   'ekf_f_2'  ]]
    labels += [['left',         'ekf left',   'right',        'ekf right']]
    linest += [['b', '--', 'r', '--']]
    fields += [['simu_lkf_1',   'ekf_f_1',    'simu_rkf_1',   'ekf_f_3'  ]]
    labels += [['left',         'ekf left',   'right',        'ekf right']]
    linest += [['b', '--', 'r', '--']]
    plot_from_logger(lgr, dt, fields, labels, 'Contact Forces', linest, ylabel=['Y [N]', 'Z [N]'])
    plut.saveFigure('contact_forces_est')

if PLOT_FORCE_VEL:
    plot_from_logger(lgr, dt, [['simu_df_'+str(i) for i in range(j,4,2)] for j in range(2)],
                     [['left', 'right']]*2, 'Force velocity', ylabel=['Y [N/s]', 'Z [N/s]'])
    plut.saveFigure('df')

if PLOT_FORCE_ACC:
    for i in range(4):
#        tmp = finite_diff(lgr.get_streams('simu_df_'+str(i)), dt)
#        lgr.add_stream(tmp, 'simu_ddf_fd_'+str(i))

        print "Max ddf", i, np.max(np.array(lgr.get_streams('simu_ddf_'+str(i))))

#        plot_ddf_err = True
#        try:
#            ddf_err = np.array(lgr.get_streams('simu_ddf_'+str(i))) - np.array(lgr.get_streams('tsid_ddf_des_'+str(i)))
#            ddf_err[int(T_DISTURB_BEGIN/dt):int(T_DISTURB_END/dt)] = 0.0
#            lgr.add_stream(ddf_err, 'tsid_ddf_err_'+str(i))
#        except:
#            plot_ddf_err = False

#    fields = [['simu_ddf_'+str(i) for i in range(j,4,2)]+['tsid_ddf_des_'+str(i) for i in range(j,4,2)]+['simu_ddf_fd_'+str(i) for i in range(j,4,2)] for j in range(2)]
#    plot_from_logger(lgr, dt, fields, [['left', 'right', 'left des', 'right des', 'left fd', 'right fd'], None],
#                     'Force acceleration', [['b', 'r', '--', '--']]*2, ylabel=['Y [N/s^2]', 'Z [N/s^2]'])
#    plut.saveFigure('ddf')
#
#    plot_from_logger(lgr, dt, [['tsid_B_ddf_max_'+str(j), 'tsid_B_ddf_UB_'+str(j), 'tsid_B_ddf_des_'+str(j)] for j in range(4)],
#                     [['max', 'UB', 'des'], None, None, None], 'Force constraint acceleration',
#                     [['b', ':', '--']]*4, ylabel=['0', '1', '2', '3'], ncols=2)

#    if(plot_ddf_err):
#        plot_from_logger(lgr, dt, [['tsid_ddf_err_'+str(i)] for i in range(4)],
#                         [['ddf err'], None, None, None], 'Force acc err',
#                         [['b']]*4, ylabel=['0', '1', '2', '3'], ncols=2)

    try:
        for i in range(4):
            tmp = finite_diff(lgr.get_streams('tsid_B_df_'+str(i)), dt)
            lgr.add_stream(tmp, 'tsid_B_ddf_fd_'+str(i))
            B_ddf_err = tmp-np.array(lgr.get_streams('tsid_B_ddf_des_'+str(i)))
            B_ddf_err[int(T_DISTURB_BEGIN/dt):int(T_DISTURB_END/dt)] = 0.0
            lgr.add_stream(B_ddf_err, 'tsid_B_ddf_err_'+str(i))

        plot_from_logger(lgr, dt, [['tsid_B_ddf_fd_'+str(i), 'tsid_B_ddf_des_'+str(i)] for i in range(4)],
                         [['fd', 'des'], None, None, None], 'Force constraint acceleration',
                         [['b', '--']]*4, ylabel=['0', '1', '2', '3'], ncols=2)

        plot_from_logger(lgr, dt, [['tsid_B_ddf_err_'+str(i)] for i in range(4)],
                         [['B*ddf err'], None, None, None], 'Force constraint acc err',
                         [['b']]*4, ylabel=['0', '1', '2', '3'], ncols=2)

        for i in range(4):
            fields = [['tsid_B_f_'+str(i)], ['tsid_B_df_'+str(i), 'tsid_B_df_max_'+str(i)], ['tsid_B_ddf_max_'+str(i), 'tsid_B_ddf_UB_'+str(i), 'tsid_B_ddf_des_'+str(i)]]
            ax = plot_from_logger(lgr, dt, fields, [['pos'], ['vel', 'vel UB'], ['max', 'UB', 'acc']], 'Force constraint '+str(i),
                                  [['-'], ['-', '--'], ['-', ':', '--']], yscale=['log','linear','linear'])
            ax[-1].set_ylim(np.min(lgr.get_streams('tsid_B_ddf_des_'+str(i))), np.max(lgr.get_streams('tsid_B_ddf_des_'+str(i))))
    except:
        pass


if PLOT_CONTACT_POINT_ACC:
    ax_lbl = {0:'Y', 1:'Z'}
    f_names = ['simu_acc_lf_', 'simu_acc_rf_', 'tsid_lf_a_des_', 'tsid_rf_a_des_']
    fields  = [[s+'0' for s in f_names]] + [[s+'1' for s in f_names]]
    labels = 2*[['real left', 'real right', 'des left', 'des right']]
    linest = 2*[['b',         'r',          '--',       '--'       ]]
    plot_from_logger(lgr, dt, fields, labels, 'Contact Point Accelerations', linest, ylabel=[s+' [m/s^2]' for s in ['Y','Z']])
    plut.saveFigure('contact_point_acc')

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
#        ylabels = [s+' '+um for (s,um) in zip(['Pos.', 'Vel.', 'Acc.', 'Jerk'], ['[m]', '[m/s]', '[m/s^2]', '[m/s^3]'])]
        ylabels = ['[m]', '[m/s]', '[m/s^2]', '[m/s^3]']
        titles  = ['Center of Mass '+ax_lbl[i]+' Pos.', 'Vel.', 'Acc.', 'Jerk']
        plot_from_logger(lgr, dt, fields, labels, titles=titles, linestyles=linest, ylabel=ylabels)
        plut.saveFigure('com_tracking_'+ax_lbl[i])

if PLOT_COM_ESTIMATION and USE_ESTIMATOR:
    ax_lbl = {0:'Y', 1:'Z'}
    for i in [0,1]:
        fields  = [['simu_com_p_'+str(i),  'tsid_com_p_est_'+str(i) ]] + [['simu_com_v_'+str(i),  'tsid_com_v_est_'+str(i) ]]
        fields += [['simu_com_a_'+str(i),  'tsid_com_a_est_'+str(i) ]] + [['simu_com_j_'+str(i),  'tsid_com_j_est_'+str(i) ]]
        labels = 4*[['real',      'estimated' ]]
        linest = 4*[[None, '--']]
        ylabels = [s+' '+um for (s,um) in zip(['Pos.', 'Vel.', 'Acc.', 'Jerk'], ['[m]', '[m/s]', '[m/s^2]', '[m/s^3]'])]
        plot_from_logger(lgr, dt, fields, labels, titles='Center of Mass '+ax_lbl[i], linestyles=linest, ylabel=ylabels)
        plut.saveFigure('com_estimate_'+ax_lbl[i])

#plot_from_logger(lgr, dt, [['tsid_dv_'+str(i), 'simu_dv_'+str(i)] for i in range(4)])

if CONTROLLER=='tsid_mistry':
    plot_from_logger(lgr, dt, [['tsid_com_j_des_'+str(i), 'tsid_com_j_exp_'+str(i), 'simu_com_j_'+str(i)] for i in range(1)], linestyles=[[None,'--',':']]*2)
    plot_from_logger(lgr, dt, [['tsid_df_des_'+str(i), 'simu_df_'+str(i)] for i in range(4)], linestyles=[[None,'--']]*4, ncols=2)
elif CONTROLLER=='tsid_flex_k' and PLOT_COM_SNAP:
    plot_from_logger(lgr, dt, [['tsid_com_s_des_'+str(i), 'tsid_com_s_exp_'+str(i), 'simu_com_s_'+str(i)] for i in range(2)], linestyles=[[None,'--',':']]*2)

if PLOT_ANGULAR_MOMENTUM_DERIVATIVES:
    plot_from_logger(lgr, dt, [['simu_am'], ['simu_dam'], ['simu_ddam']])
    plut.saveFigure('angular_momentum')

if(PLOT_JOINT_TORQUES):
    labels = [['right hip des', 'right knee des', 'left hip des', 'left knee des']+['right hip', 'right knee', 'left hip', 'left knee']]
    plot_from_logger(lgr, dt, [['tsid_tau_'+str(i) for i in range(4)] + ['simu_tauq_'+str(i) for i in range(3,7)]], 
                                labels, linestyles=[4*['-']+4*['--']], ylabel='Joint Generalized Force [N]/[Nm]')
    plut.saveFigure('joint_torques')

if(PLOT_JOINT_ANGLES):
    plot_from_logger(lgr, dt, [['simu_q_'+str(i), 'tsid_q_cmd_'+str(i-4)] for i in range(4,8)], [['q', 'q_cmd']]*4,
                          titles=['right hip', 'right knee', 'left hip', 'left knee'],
                          linestyles=[['-', '--']]*4, ncols=2, ylabel='Joint Angles [rad]]')
    plut.saveFigure('joint_angles')

if(SAVE_DATA):
    lgr.dump_compressed(RESULTS_PATH+'logger_data')
#tfile = open(plot_utils.FIGURE_PATH+conf.TEXT_FILE_NAME, "w")
#tfile.write(info);
#tfile.close();

if(SHOW_FIGURES):
    plt.show()
