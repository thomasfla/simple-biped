import sys
import numpy as np
import numpy.linalg as la
import numpy.ma as ma
from scipy.linalg import expm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numpy.linalg import eigvals
from numpy.random import uniform
from math import sqrt
from sea_dynamics import SEA, get_SEA_parameters, ImpedanceControl
from sea_dynamics_closed_loop import CloseLoopSEA, CloseLoopDtSEA

np.set_printoptions(precision=2, suppress=True, linewidth=100);

def compute_lambda_bounds(dt, k, I_j, b_j, I_m, b_m, k_tau_bounds, b_tau_bounds, n_points, use_ff, log_space=True):
    if(log_space):
        k_tau = np.logspace(np.log10(k_tau_bounds[0]), np.log10(k_tau_bounds[1]), n_points)
        b_tau = np.logspace(np.log10(b_tau_bounds[0]), np.log10(b_tau_bounds[1]), n_points)
    else:
        k_tau = np.linspace(k_tau_bounds[0], k_tau_bounds[1], n_points)
        b_tau = np.linspace(b_tau_bounds[0], b_tau_bounds[1], n_points)
    K,B = np.meshgrid(k_tau, b_tau)
    lambda_min = np.empty((n_points, n_points))
    lambda_max = np.empty((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            if(dt<=0.0):
                cl_sea = CloseLoopSEA(1.0, k, I_j, b_j, I_m, b_m, 1.0, 1.0, K[i,j], B[i,j])
            else:
                cl_sea = CloseLoopDtSEA(dt, k, I_j, b_j, I_m, b_m, 1.0, 1.0, K[i,j], B[i,j], use_ff)
            [lambda_min[i,j], lambda_max[i,j]] = cl_sea.compute_lambda_bounds()
    
    # mask nan
    lambda_min = ma.masked_invalid(lambda_min)
    lambda_max = ma.masked_invalid(lambda_max)
    
    return (K, B, lambda_min, lambda_max)
    
# USER PARAMETERS
COMPUTE_LAMBDA_BOUNDS = 1
TEST_LAMBDA_BOUNDS = -10000
TEST_ROUTH_HURWITZ = -10000
SAMPLE_STABLE_IMPEDANCE_SPACE = -2000
TEST_CLOSED_LOOP_SEA = -10

dt = 1e-3
T = 1.5
x_0 = np.array([0.1, 0.0, 0.1, 0.0])
USE_TORQUE_FF = True

SEA_param_name = 'Paine2017'
#SEA_param_name = 'Focchi2013'
(k, I_j, b_j, I_m, b_m) = get_SEA_parameters(SEA_param_name)

MAX_KP = I_j*1e5
MAX_KD = 3*sqrt(MAX_KP)
KTAU_BOUNDS = [1e-7, 1e3]
BTAU_BOUNDS = [1e-4, 1e0]
N_TESTS = int(1e2)

if(SEA_param_name == 'Focchi2013'):
    #(k_tau_0, b_tau_0) = (0.1, 0.013)  # lambda_max found:   339.6
    (k_tau_0, b_tau_0) = (1.0, 0.04)   # lambda_max found:   134.7
    #(k_tau_0, b_tau_0) = (1e2, 0.37)    # lambda_max found:   279.8
    KTAU_BOUNDS = [1e-2, 1e3]
    BTAU_BOUNDS = [1e-2, 1e0]
elif(SEA_param_name == 'Paine2017'):
    (k_tau_0, b_tau_0) = (0.1, 0.1)
    if(dt<=0.0):
        KTAU_BOUNDS = [1e-7, 1e10]
        BTAU_BOUNDS = [1e-4, 1e4]
    else:
        KTAU_BOUNDS = [1e-3, 1e0]
        BTAU_BOUNDS = [1e-5, 1e-3]


print "Gonna use SEA parameters from", SEA_param_name
print "SEA stiffness is", k

if(COMPUTE_LAMBDA_BOUNDS):
    print "\n******************* COMPUTE LAMBDA BOUNDS *******************"
    print "Assuming kp=lambda^2 and kd=2*lambda, compute the lambda bounds"
    print "for different values of the torque PD gains."
    print "We sample k_tau in [%f, %f]"%(KTAU_BOUNDS[0], KTAU_BOUNDS[1])
    print "We sample b_tau in [%f, %f]"%(BTAU_BOUNDS[0], BTAU_BOUNDS[1])
    
    (K, B, l_min, l_max) = compute_lambda_bounds(dt, k, I_j, b_j, I_m, b_m, KTAU_BOUNDS, BTAU_BOUNDS, 50, USE_TORQUE_FF)
    
    # compute lambda bounds for zero torque PD gains
    if(dt<=0.0):
        cl_sea = CloseLoopSEA(1.0, k, I_j, b_j, I_m, b_m, 1.0, 1.0, 0.0, 0.0)
    else:
        cl_sea = CloseLoopDtSEA(dt, k, I_j, b_j, I_m, b_m, 1.0, 1.0, 0.0, 0.0)
    [l_min_0, l_max_0] = cl_sea.compute_lambda_bounds()
    print "For k_tau=b_tau=0 lambda_max is %7.1f"%(l_max_0)
    
    i = np.unravel_index(np.argmax(l_max), l_max.shape)
    print "Largest  lambda_max found: %7.1f for k_tau=%5f, b_tau=%5f" % (
        l_max[i], K[i], B[i])
    i = np.unravel_index(np.argmax(l_min), l_max.shape)
    print "Largest  lambda_min found: %7.1f for k_tau=%5f, b_tau=%5f" % (
        l_min[i], K[i], B[i])
    i = np.unravel_index(np.argmin(l_max), l_max.shape)
    print "Smallest lambda_max found: %7.1f for k_tau=%5f, b_tau=%5f" % (
        l_max[i], K[i], B[i])
    i = np.unravel_index(np.argmin(l_min), l_max.shape)
    print "Smallest lambda_min found: %7.1f for k_tau=%5f, b_tau=%5f" % (
        l_min[i], K[i], B[i])
        
    # plot using different scales
    plot_scales = [('log', 'log')]
    for ps in plot_scales:
        fig = plt.figure(figsize = (12,8))
        fig.subplots_adjust(wspace=0.3)
        if(ps[0]=='log'):
            # use log scale for color map
            plt.pcolormesh(K, B, l_max, cmap=plt.cm.get_cmap('Blues'),
                           norm=colors.LogNorm(vmin=l_max.min(), vmax=l_max.max()))
        else:
            plt.pcolormesh(K, B, l_max, cmap=plt.cm.get_cmap('Blues'))
        plt.colorbar()
        
        # draw a red cross over optimal point
        i = np.unravel_index(np.argmax(l_max), l_max.shape)
        plt.plot(K[i], B[i], 'ro', markersize=12)
        
        plt.xlabel('k_tau')
        plt.ylabel('b_tau')
        plt.title('Lambda Max')
        if(ps[1]=='log'):
            plt.xscale('log')
            plt.yscale('log')
    plt.show()


if(TEST_LAMBDA_BOUNDS>0):
    print "\n\n******************* TEST LAMBDA BOUNDS *******************"
    print "Gonna perform %d tests on lambda bounds to check that" % (TEST_LAMBDA_BOUNDS)
    print "when lambda is within the bounds the closed-loop system is stable."
    
    n_err = 0;
    n_unstable = 0;
    for i in range(TEST_LAMBDA_BOUNDS):
        # sample random gains
        k_p   = uniform(low=0.0, high=MAX_KP)
        lmbda = sqrt(k_p)
        k_d   = 2*lmbda
        k_tau = uniform(low=KTAU_BOUNDS[0], high=KTAU_BOUNDS[1])
        b_tau = uniform(low=BTAU_BOUNDS[0], high=BTAU_BOUNDS[1])
    
        cl_sea = CloseLoopSEA(dt, k, I_j, b_j, I_m, b_m, k_p, k_d, k_tau, b_tau)
        maxEv = cl_sea.max_eigen_value()
        [lambda_min, lambda_max] = cl_sea.compute_lambda_bounds()
    
        if(maxEv>=0.0):
            n_unstable += 1;
            if(lmbda>lambda_min and lmbda<lambda_max):
                print "[ERROR] System unstable but lambda=%.2f in [%.1f, %.1f]"%(
                        lmbda, lambda_min, lambda_max)
        elif(lmbda<lambda_min or lmbda>lambda_max):
                print "[ERROR] System stable but lambda=%.2f not in [%.1f, %.1f]"%(
                        lmbda, lambda_min, lambda_max)
    print "Tests finished. Number of unstable gains found: %d" % (n_unstable)
    print "Number of errors found: %d" %(n_err)
    

if(TEST_ROUTH_HURWITZ>0):
    print "\n\n******************* TEST ROUTH_HURWITZ *******************"
    print "Gonna perform %d tests on Routh-Hurwitz conditions to check that" % (TEST_ROUTH_HURWITZ)
    print "when they are satisfied the closed-loop system is stable."

    n_err = 0;
    n_unstable = 0;
    for i in range(TEST_ROUTH_HURWITZ):
        # sample random gains
        k_p   = uniform(low=0.0, high=MAX_KP)
        k_d   = uniform(low=0.0, high=MAX_KD)
        k_tau = uniform(low=KTAU_BOUNDS[0], high=KTAU_BOUNDS[1])
        b_tau = uniform(low=BTAU_BOUNDS[0], high=BTAU_BOUNDS[1])
    
        cl_sea = CloseLoopSEA(dt, k, I_j, b_j, I_m, b_m, k_p, k_d, k_tau, b_tau)
        maxEv = cl_sea.max_eigen_value()
        stableRH = cl_sea.check_RH_conditions()
    
        if(maxEv>=0.0):
            n_unstable += 1;
    
        if((maxEv<0.0 and not stableRH) or (maxEv>=0.0 and stableRH)):
            n_err += 1;
            print "[ERROR] gains:", k_p, k_d, k_tau, b_tau
            print "        Routh-Hurtwitz says", stableRH, ", but max eigenvalue is", maxEv
   
    print "Test finished. Number of unstable gains found: %d" % (n_unstable)
    print "%d errors out of %d tests" %(n_err, TEST_ROUTH_HURWITZ)
    

if(SAMPLE_STABLE_IMPEDANCE_SPACE>0):
    print "\n\n******************* SAMPLE STABLE IMPEDANCE SPACE *******************"
    print "Gonna take %d samples of impedance space using constant" % (SAMPLE_STABLE_IMPEDANCE_SPACE)
    print "torque PD gain. k_tau=%f, b_tau=%f"%(k_tau_0, b_tau_0)
    print "and sampling kp in [0.0, %.1f]"%(MAX_KP)
    print "and sampling kd in [0.0, %.1f]"%(MAX_KD)
    print "For the discrete-time case dt=%f"%(dt)
    
    kp_unstable = [];
    kd_unstable = [];
    kp_stable = []
    kd_stable = []
    kp_unstable_dt = [];
    kd_unstable_dt = [];
    kp_stable_dt = []
    kd_stable_dt = []
    for i in range(SAMPLE_STABLE_IMPEDANCE_SPACE):
        # sample random gains
        k_p   = uniform(low=0.0, high=MAX_KP)
        k_d   = uniform(low=0.0, high=MAX_KD)
        cl_ct_sea = CloseLoopSEA(dt, k, I_j, b_j, I_m, b_m, k_p, k_d, k_tau_0, b_tau_0, USE_TORQUE_FF)
        cl_dt_sea = CloseLoopDtSEA(dt, k, I_j, b_j, I_m, b_m, k_p, k_d, k_tau_0, b_tau_0, USE_TORQUE_FF)
        
        maxEvCt = cl_ct_sea.max_eigen_value()
        if(maxEvCt>=0.0):
            kp_unstable += [k_p,]
            kd_unstable += [k_d,]
        else:
            kp_stable += [k_p,]
            kd_stable += [k_d,]
            
        maxEvDt = cl_dt_sea.max_eigen_value()
        if(maxEvDt>=1.0):
            kp_unstable_dt += [k_p,]
            kd_unstable_dt += [k_d,]
        else:
            kp_stable_dt += [k_p,]
            kd_stable_dt += [k_d,]
            
    print "Sampling finished."
    
    (l_min, l_max) = cl_dt_sea.compute_lambda_bounds()
    print "Max stable kp=", l_max**2
    
    pl = [(kp_stable,    kd_stable,    kp_unstable,    kd_unstable, 'Continuous Time'),
          (kp_stable_dt, kd_stable_dt, kp_unstable_dt, kd_unstable_dt, 'Discrete Time')]

    for (kps,kds,kpu,kdu,title) in pl:
        fig = plt.figure(figsize = (9,9))
        plt.plot(kpu, kdu, 'r o', label='unstable')
        plt.plot(kps, kds,   'k o', label='stable')
        kp = np.arange(0.0, MAX_KP, 1.0)
        kd = 2.0*np.sqrt(kp)
        plt.plot(kp, kd, label='kd=2*sqrt(kp)')
        plt.legend()
        plt.xlabel('Kp')
        plt.ylabel('Kd')
        plt.title(title)
    plt.show()
    
    
if(TEST_CLOSED_LOOP_SEA>0):
    print "\n\n******************* TEST CLOSED-LOOP SEA *******************"
    print "Gonna perform %d tests to check that closed-loop SEA dynamics" % (TEST_CLOSED_LOOP_SEA)
    print "matches open-loop SEA dynamics coupled with discrete-time controller"
    print "with dt=%f"%(dt)

    DO_PLOTS = 0
    USE_DISCRETE_TIME_CLOSED_LOOP = 1
    N = int(T/dt)
    sea    = SEA(dt, k, I_j, b_j, I_m, b_m)
    n_unstable = 0;
    k_tau = k_tau_0
    b_tau = b_tau_0
    for i in range(TEST_CLOSED_LOOP_SEA):
        # sample random gains
        k_p   = uniform(low=0.0, high=MAX_KP)
        k_d   = uniform(low=0.0, high=MAX_KD)
#        k_tau = uniform(low=KTAU_BOUNDS[0], high=KTAU_BOUNDS[1])
#        b_tau = uniform(low=BTAU_BOUNDS[0], high=BTAU_BOUNDS[1])
    
        ctrl = ImpedanceControl(k_p, k_d, k_tau, b_tau, k, USE_TORQUE_FF)
        if(USE_DISCRETE_TIME_CLOSED_LOOP):
            cl_sea = CloseLoopDtSEA(dt, k, I_j, b_j, I_m, b_m, k_p, k_d, k_tau, b_tau)
        else:
            cl_sea = CloseLoopSEA(dt, k, I_j, b_j, I_m, b_m, k_p, k_d, k_tau, b_tau)
            
        if(not cl_sea.is_stable()):
            n_unstable += 1
            continue
    
        # initialize system state
        sea.x    = x_0
        cl_sea.x = x_0
    
        x    = np.zeros((N,4))
        x_cl = np.zeros((N,4))
        u = np.zeros(N)
        #tau_d = np.zeros(N)
        x[0,:] = x_0
        x[0,2] = sea.tau()
        x[0,3] = sea.dtau()
        x_cl[0,:] = x_0
        x_cl[0,2] = cl_sea.tau()
        x_cl[0,3] = cl_sea.dtau()
        for t in range(N-1):
            #tau_d[t] = -k_p*sea.qj() -k_d*sea.dqj()
            #u[t] = -k_tau*(sea.tau() - tau_d[t]) - b_tau*sea.dtau()
            u[t] = ctrl.get_u(sea.x)
            
            x[t+1,:] = sea.simulate(u[t])
            x[t+1,2] = sea.tau()
            x[t+1,3] = sea.dtau()
        
            x_cl[t+1,:] = cl_sea.simulate()
            x_cl[t+1,2] = cl_sea.tau()
            x_cl[t+1,3] = cl_sea.dtau()
        
        print "Test %d. Max position traj error: "%(i), np.max(np.abs(x[:,0]-x_cl[:,0]))
        
        if(DO_PLOTS):
            f, ax = plt.subplots(5,1,sharex=True);
            labels = ['qj','dqj','tau','dtau']
            for j in range(4):
                ax[j].plot(x[:,j],    '-',  label=labels[j]+' open loop');
                ax[j].plot(x_cl[:,j], 'r--', label=labels[j]+' closed loop');
                #if(j==2):
                #    ax[j].plot(tau_d, label='tau_des')
                ax[j].legend()
            ax[-1].plot(u, label='tau_m')
            ax[-1].legend()
            plt.show();
    
    print "Test finished. Number of unstable gains found: %d" % (n_unstable)
