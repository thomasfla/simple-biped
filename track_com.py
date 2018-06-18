import pinocchio as se3
from pinocchio.utils import *
from math import pi,sqrt
from hrp2_reduced import Hrp2Reduced
import time 
from simu import Simu, ForceDict
from utils.utils_thomas import restert_viewer_server, traj_sinusoid, finite_diff
from utils.logger import RaiLogger
from utils.filters import FIR1LowPass, BALowPass, FiniteDiff
import matplotlib.pyplot as plt
from tsid import Tsid
from tsid_flexible_contacts import TsidFlexibleContact
from path import pkg, urdf 
from utils.noise_utils import NoisyState
from estimators import Kalman, get_com_and_derivatives

from estimation.momentumEKF import *
from utils.plot_utils import plot_gain_stability
import os

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

#    dv      = tsid.data.dv
#    tau_des = tsid.data.tau    
    #estimate actual torque from (filtered) contact forces
#    f_filtered = FTSfilter.update(f)
#    tau_est  = compute_torques_from_dv_and_forces(dv,f_filtered)
#    tau_est_fd  = dtau_fd_filter.update(tau_est) 
#    dtau_est = dtau_lp_filter.update(tau_est_fd)
    #~ tau_ctrl = torque_controller(tau_des,tau_est)

#    if not FLEXIBLE_CONTROLLER:
        #~ tau_ctrl = tau_des + Ktau*(tau_des-tau_est) - Kdtau * dtau_est
        # regulation on forces
        #~ f_des = tsid.data.f 
        #~ f_ctrl = f_des + Ktau * Ktau * (f_des - f_filtered)
        #~ tau_ctrl = compute_torques_from_dv_and_forces(dv,f_ctrl) - Kdtau * df
#        tau_ctrl = tau_des 
#    else:
#        tau_ctrl = tau_des 

#    lgr.add_vector(tau_ctrl,   'tau_ctrl')
#    lgr.add_vector(tau_est,    'tau_est')
#    lgr.add_vector(tau_est_fd, 'tau_est_fd')
#    lgr.add_vector(dtau_est,   'dtau_est')
    
    if not log_index%100 :
        print "t:{0} \t com error \t{1} ".format(log_index*dt, np.linalg.norm(tsid.data.com_p_err))
        
    # check that the state doesn't go crazy
    if np.linalg.norm(tsid.data.com_p_err) > 0.1:
        raise ValueError("COM error > 0.1")
    if np.linalg.norm(tsid.data.f) > 500:
        raise ValueError("Forces > 500")
    if np.linalg.norm(q) > 10:
        raise ValueError("q > 10")
        
    return np.vstack([f_disturb_traj(t), tsid.data.tau]) #tau_ctrl])
    
def f_disturb_traj(t):
    if DISTURB:
        if (t>0.5 and t<0.8 ):
            return np.matrix([30.,0.,0.]).T
        #~ if (t>0.9 and t<0.91 ):
            #~ return np.matrix([300.,0.,0.]).T
    return np.matrix([0.,0.,0.]).T
    
def com_traj(t, c_init, c_final, T):
    py,vy,ay,jy,sy = traj_sinusoid(t, c_init[0], c_final[0], T)
    pz,vz,az,jz,sz = traj_sinusoid(t, c_init[1], c_final[1], T)
    return (np.matrix([[py ],[pz]]), np.matrix([[vy ],[vz]]), 
            np.matrix([[ay ],[az]]), np.matrix([[jy ],[jz]]), np.matrix([[sy ],[sz]]))
    
def compute_torques_from_dv_and_forces(dv,f):
    M  = robot.data.M        #(7,7)
    h  = robot.data.nle      #(7,1)
    Jl,Jr = robot.get_Jl_Jr_world(q, False)
    Jc = np.vstack([Jl[1:3],Jr[1:3]])    # (4, 7)
    tau = (M*dv - Jc.T*f + h)[3:]
    return tau
    

robot = Hrp2Reduced(urdf,[pkg],loadModel=True,useViewer=useViewer)
robot.display(robot.q0)
if useViewer:
    robot.viewer.setCameraTransform(0,[1.9154722690582275, -0.2266872227191925, 0.1087859719991684,
                                       0.5243823528289795, 0.518651008605957, 0.4620114266872406, 0.4925136864185333])

#Plots
PLOT_COM_AND_FORCES = True
PLOT_COM_DERIVATIVES = True   
PLOT_ANGULAR_MOMENTUM_DERIVATIVES = False   
   
#Simulation parameters
dt  = 1e-3
ndt = 5
simulation_time = 2.0
USE_REAL_STATE = 0       # use real state for controller feedback
#robot parameters
tauc = 0.*np.array([1.,1.,1.,1.])#coulomb friction
Ky = 23770.
Kz = 239018.
By = 50. *0.
Bz = 500.*0.
Kspring = -np.diagflat([Ky,Kz,0.])     # Stiffness of the feet spring
Bspring = -np.diagflat([By,Bz,0.])     # damping of the feet spring
g = 9.81                                # gravity acceleration

#Controller parameters
fc_dtau_filter = 100.           # cutoff frequency of the filter applyed to the finite differences of the torques 
FLEXIBLE_CONTROLLER = True      # if True it uses the controller for flexible contacts
DISTURB = 1                     # if True disturb the motion with an external force
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
COM_REF_START = [0.0, 0.53]
COM_REF_END   = [0.0, 0.53]
COM_TRAJ_TIME = 2.0

#Grid of gains to try:
Kp_coms = []    # np.linspace(1,100,50)
Kd_coms = []    # np.linspace(1,500,50)

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

se3.computeAllTerms(robot.model,robot.data,q0,v0)
m = robot.data.mass[0] 
q0[1]-=0.5*m*g/Kz
f0,df0 = simu.compute_f_df_from_q_v(q0,v0)
c0,dc0,ddc0,dddc0 = get_com_and_derivatives(robot,q0,v0,f0,df0)
l0 = 0

dtau_fd_filter = FiniteDiff(dt)
dtau_lp_filter = FIR1LowPass(np.exp(-2*np.pi*fc_dtau_filter*dt)) # Force sensor filter
FTSfilter      = FIR1LowPass(np.exp(-2*np.pi*fc*dt))             # Force sensor filter

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
centroidalEstimator = MomentumEKF(dt, m, g, c0.A1, dc0.A1, np.array([l0]), f0.A1, S_0, sigma_c, sigma_dc, sigma_l, sigma_f, sigma_ddf)

#~ estimator = None
#~ b, a = np.array ([0.00554272,  0.01108543,  0.00554272]), np.array([1., -1.77863178,  0.80080265])
#~ FTSfilter = BALowPass(b,a,"butter_lp_filter_Wn_05_N_2")  

last_vlf = np.matrix([0,0,0]).T;
last_vrf = np.matrix([0,0,0]).T;
log_size = int(simulation_time / dt)    #max simulation samples
log_t        = np.zeros([log_size,1])+np.nan  # time
log_index = 0  

if FLEXIBLE_CONTROLLER:
    # Kp_com, Kd_com, Ka_com, Kj_com = 1, 2.7, 3.4, 2.1
    # Kp_post = 1e-6
    (Kp_com, Kd_com, Ka_com, Kj_com) = (1.20e+06, 1.54e+05, 7.10e+03, 1.40e+02)
    #~ Kp_com,Kd_com,Ka_com,Kj_com = 2.4e+09, 5.0e+07, 3.5e+05, 1.0e+03
    #~ Kp_com,Kd_com,Ka_com,Kj_com = 17160.0,  6026.0,   791.0,    46.  
    tsid=TsidFlexibleContact(robot,Ky,Kz,w_post,Kp_post,Kp_com, Kd_com, Ka_com, Kj_com, estimator)
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

lgr.auto_log_local_variables(['com_p', 'com_v', 'com_a', 'com_j'], [vc, vc, vc, vc])
lgr.auto_log_local_variables(['f'], [vc], log_var_names=[['lkf_sensor_0', 'lkf_sensor_1', 'rkf_sensor_0', 'rkf_sensor_1']])
lgr.auto_log_local_variables(['df'], [vc], log_var_names=[['lkdf_sensor_0', 'lkdf_sensor_1', 'rkdf_sensor_0', 'rkdf_sensor_1']])
    
def loop(q, v, f, df, niter, ndt=None, dt=None, tsleep=.9, fdisplay=100):
    global log_index, last_vlf, last_vrf
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
        # centroidalEstimator.predict_update(com_noisy.A1, com_v_noisy.A1, np.array([l_noisy]), f_noisy, p, ddf)
        
        # simulate the system
        u = control(q_noisy,v_noisy,f_noisy,df_noisy)
        q,v,f,df = simu(q,v,u)

        # log data        
        log_t[log_index] = log_index*simu.dt
        com_p, com_v, com_a, com_j = get_com_and_derivatives(robot,q,v,f,df)
        lgr.log_all(locals())
        
        lgr.add_vector((simu.vlf-last_vlf).A1/simu.dt, 'a_lf_fd')   # feet accelerations via finite differences
        lgr.add_vector((simu.vrf-last_vrf).A1/simu.dt, 'a_rf_fd')
        last_vlf, last_vrf = simu.vlf, simu.vrf
        
        # get feet acceleration via jacobian.
        Jl,Jr = robot.get_Jl_Jr_world(q)
        driftLF,driftRF = robot.get_driftLF_driftRF_world(q,v)
        Mlf,Mrf = robot.get_Mlf_Mrf(q)
        lgr.add_vector(Mlf.translation[1:3].A1, 'p_lf')
        lgr.add_vector(Mrf.translation[1:3].A1, 'p_rf')        

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
        tsid.Kp_com, tsid.Kd_com = Kp_com, Kd_com
        
        #reset all entity with internal states
        simu.reset()
        FTSfilter.reset()
        dtau_fd_filter.reset()
        dtau_lp_filter.reset()
        
        #start from initial state
        q,v,f,df = q0.copy(), v0.copy(), f0.copy(), df0.copy()
        isstable = True
        
        #simulate and test stability
        try:
            q,v = loop(q,v,f,df,log_size)
        except ValueError:
            isstable=False

        print "Kp_com={}, Kd_com={}, Stable? {}".format(Kp_com, Kd_com, isstable)
        KpKd.append([Kp_com,Kd_com])
        stab_grid[i,j] = isstable
        Kp_grid[i,j], Kd_grid[i,j] = Kp_com, Kd_com
        
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
    plt.text(0.1, 0.05, infostr,
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.title('com tracking Y')
    plt.plot(log_t, lgr.tsid_com_p_mes_0, label="measured com")
    plt.plot(log_t, lgr.tsid_com_p_err_0, label="com error")
    plt.plot(log_t, lgr.tsid_comref_0,    label="com ref")
    plt.legend()
    plt.subplot(312, sharex=ax1)
    plt.title('feet forces Y') 
    plt.plot(log_t, lgr.get_streams('lkf_sensor_0'),label="applied force",color='b')
    plt.plot(log_t, lgr.get_streams('rkf_sensor_0'),label="applied force",color='r')
    plt.plot(log_t, lgr.tsid_lkf_0, label="force TSID",color='b',linestyle=':')
    plt.plot(log_t, lgr.tsid_rkf_0, label="force TSID",color='r',linestyle=':')
    plt.legend()
    plt.subplot(313, sharex=ax1)
    plt.title('feet forces Z')
    plt.plot(log_t, lgr.get_streams('lkf_sensor_1'),label="applied force",color='b')
    plt.plot(log_t, lgr.get_streams('rkf_sensor_1'),label="applied force",color='r')
    plt.plot(log_t, lgr.tsid_lkf_1, label="force TSID",color='b',linestyle=':')
    plt.plot(log_t, lgr.tsid_rkf_1, label="force TSID",color='r',linestyle=':')
    plt.legend()
    plt.show()


if PLOT_COM_DERIVATIVES :
    axe_label = {0:'Y', 1:'Z'}    
    for i in [0,1]:
        ax1 = plt.subplot(511)
        plt.plot(log_t, lgr.get_streams('tsid_com_p_mes_'+str(i)), label="measured com "+axe_label[i])
        plt.plot(log_t, lgr.get_streams('tsid_com_p_est_'+str(i)), label="estimated com "+axe_label[i])
        plt.plot(log_t, lgr.get_streams('com_p_'+str(i)), label="com " + axe_label[i]) 
        plt.legend()
        plt.subplot(512,sharex=ax1)
        plt.plot(log_t, lgr.get_streams('tsid_com_v_mes_'+str(i)), label="measured vcom "+ axe_label[i]) 
        plt.plot(log_t, lgr.get_streams('tsid_com_v_est_'+str(i)), label="estimated vcom "+ axe_label[i]) 
        plt.plot(log_t, lgr.get_streams('com_v_'+str(i)), label="vcom "+ axe_label[i]) 
        plt.plot(log_t, finite_diff(lgr.get_streams('com_p_'+str(i)), dt),':', label="fd com " + axe_label[i]) 
        plt.legend()
        plt.subplot(513,sharex=ax1)
        plt.plot(log_t, lgr.get_streams('tsid_com_a_mes_'+str(i)), label="measured acom "+ axe_label[i]) 
        plt.plot(log_t, lgr.get_streams('tsid_com_a_est_'+str(i)), label="estimated acom "+ axe_label[i]) 
        if not FLEXIBLE_CONTROLLER :
            plt.plot(log_t, lgr.get_streams('tsid_com_a_des_'+str(i)), label="desired acom" + axe_label[i])
        plt.plot(log_t, lgr.get_streams('com_a_'+str(i)), label="acom "+ axe_label[i]) 
        plt.plot(log_t, finite_diff(lgr.get_streams('com_v_'+str(i)),dt), ':', label="fd vcom " + axe_label[i]) 
        plt.legend()
        plt.subplot(514,sharex=ax1)
        plt.plot(log_t, lgr.get_streams('tsid_com_j_est_'+str(i)), label="estimated jcom "+ axe_label[i]) 
        plt.plot(log_t, lgr.get_streams('com_j_'+str(i)), label="jcom "+ axe_label[i]) 
        plt.plot(log_t, finite_diff(lgr.get_streams('com_a_'+str(i)), dt),':', label="fd acom " + axe_label[i])
        plt.legend()
        plt.subplot(515,sharex=ax1)
        if FLEXIBLE_CONTROLLER :
            plt.plot(log_t, lgr.get_streams('tsid_com_s_des_'+str(i)), label="desired scom" + axe_label[i])
        plt.plot(log_t, finite_diff(lgr.get_streams('com_j_'+str(i)), dt), ':', label="fd jcom " + axe_label[i])
        #~ plt.plot(log_t,log_lkf_sensor[:,1], label="force Left z")
        #~ plt.plot(log_t,log_rkf_sensor[:,1], label="force Right z")
        plt.legend()
        plt.show()

if PLOT_ANGULAR_MOMENTUM_DERIVATIVES:
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

#~ plt.plot(log_a_lf_fd, label = "a_lf_fd")
#~ plt.plot(log_a_rf_fd, label = "a_rf_fd")
#~ plt.plot(log_a_lf_jac, label = "a_lf_jac")
#~ plt.plot(log_a_rf_jac, label = "a_rf_jac")

#log_a_lf_fd[0]=np.nan
#log_a_rf_fd[0]=np.nan
#plt.plot(log_a_lf_fd[:,:2], label = "log_a_lf_fd")
#plt.plot(log_a_lf_jac[:,:2], label = "a_lf_jac")
#plt.plot(log_a_lf_des, label = "log_a_lf_des")
#plt.legend()
#plt.show()

#plt.plot(log_dv_simu[:] - log_dv_tsid[:], label = "dv_simu-dv_tsid")
#plt.legend()
#plt.show()

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
try:
    embed()
except:
    pass;
