import sys
import numpy as np
import numpy.linalg as la
from scipy.linalg import expm
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
from numpy.random import uniform
from math import sqrt
from sea_dynamics import SEA
from sea_dynamics_closed_loop import CloseLoopSEA

np.set_printoptions(precision=2, suppress=True, linewidth=100);

def compute_lambda_bounds(k, I_j, b_j, I_m, b_m, k_tau_bounds, b_tau_bounds, n_points, log_space=True):
    if(log_space):
        k_tau = np.logspace(np.log10(k_tau_bounds[0]), np.log10(k_tau_bounds[1]), n_points)
        b_tau = np.logspace(np.log10(b_tau_bounds[0]), np.log10(b_tau_bounds[1]), n_points)
    else:
        k_tau = np.linspace(k_tau_bounds[0], k_tau_bounds[1], n_points)
        b_tau = np.linspace(b_tau_bounds[0], b_tau_bounds[1], n_points)
    K,B = np.meshgrid(k_tau, b_tau)
    print 'k_tau', k_tau
    print 'b_tau', b_tau
    lambda_min = np.empty((n_points, n_points))
    lambda_max = np.empty((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):    
            cl_sea = CloseLoopSEA(1.0, k, I_j, b_j, I_m, b_m, 1.0, 1.0, K[i,j], B[i,j])
            [lambda_min[i,j], lambda_max[i,j]] = cl_sea.compute_lambda_bounds()
    
    return (K, B, lambda_min, lambda_max)
    
# USER PARAMETERS
COMPUTE_LAMBDA_BOUNDS = 1
TEST_LAMBDA_BOUNDS = 1000
TEST_ROUTH_HURWITZ = 1000
SAMPLE_STABLE_IMPEDANCE_SPACE = 10000
TEST_CLOSED_LOOP_SEA = 10

dt = 1e-5
T = 1.5
x_0 = np.array([0.1, 0.0, 0.1, 0.0])

k = 1e5         # spring stiffness
I_j = 0.014     # joint inertia
b_j = 0.1       # joint damping
I_m = 0.225     # motor inertia
b_m = 0.1*1.375     # motor damping

k_tau_0 = 0.1 #01*30.0;
b_tau_0 = 0.1 #*0.57;
k_p = 100.0*I_j;
k_d = 2.0*sqrt(k_p)

MAX_KP = I_j*1e7
MAX_KD = 3*sqrt(MAX_KP) #I_j*1e6
KTAU_BOUNDS = [0.1, 1e6]
BTAU_BOUNDS = [0.1, 10.0]
N_TESTS = int(1e2)

if(COMPUTE_LAMBDA_BOUNDS):
    print "\n******************* COMPUTE LAMBDA BOUNDS *******************"
    
    (K, B, l_min, l_max) = compute_lambda_bounds(k, I_j, b_j, I_m, b_m, KTAU_BOUNDS, BTAU_BOUNDS, 50)
    
    i = np.unravel_index(np.argmax(l_max), l_max.shape)
    print "Largest  lambda_max found: %.1f for k_tau=%.1f, b_tau=%.1f" % (
        l_max[i], K[i], B[i])
    i = np.unravel_index(np.argmax(l_min), l_max.shape)
    print "Largest  lambda_min found: %.1f for k_tau=%.1f, b_tau=%.1f" % (
        l_min[i], K[i], B[i])
    i = np.unravel_index(np.argmin(l_max), l_max.shape)
    print "Smallest  lambda_max found: %.1f for k_tau=%.1f, b_tau=%.1f" % (
        l_max[i], K[i], B[i])
    i = np.unravel_index(np.argmin(l_min), l_max.shape)
    print "Smallest  lambda_min found: %.1f for k_tau=%.1f, b_tau=%.1f" % (
        l_min[i], K[i], B[i])
        
    fig = plt.figure(figsize = (12,8))
    fig.subplots_adjust(wspace=0.3)
    plt.pcolormesh(K, B, l_max, cmap=plt.cm.get_cmap('Blues'))
    plt.colorbar()
    plt.xlabel('k_tau')
    plt.ylabel('b_tau')
    plt.title('Lambda Max')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


if(TEST_LAMBDA_BOUNDS>0):
    print "\n******************* TEST LAMBDA BOUNDS *******************"
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
    print "Number of errors found: %d\n" %(n_err)
    

if(TEST_ROUTH_HURWITZ>0):
    print "\n******************* TEST ROUTH_HURWITZ *******************"
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
    print "\n******************* SAMPLE STABLE IMPEDANCE SPACE *******************"
    print "Gonna take %d samples of impedance space using constant" % (SAMPLE_STABLE_IMPEDANCE_SPACE)
    print "torque PD gain. k_tau=%.1f\t b_tau=%.1f"%(k_tau_0, b_tau_0)
    print "and sampling kp in [0.0, %.1f]"%(MAX_KP)
    print "and sampling kd in [0.0, %.1f]"%(MAX_KD)
    
    kp_unstable = [];
    kd_unstable = [];
    kp_good = []
    kd_good = []
    for i in range(SAMPLE_STABLE_IMPEDANCE_SPACE):
        # sample random gains
        k_p   = uniform(low=0.0, high=MAX_KP)
        k_d   = uniform(low=0.0, high=MAX_KD)
        cl_sea = CloseLoopSEA(dt, k, I_j, b_j, I_m, b_m, k_p, k_d, k_tau_0, b_tau_0)
        maxEv = cl_sea.max_eigen_value()
        if(maxEv>=0.0):
            kp_unstable += [k_p,]
            kd_unstable += [k_d,]
        else:
            kp_good += [k_p,]
            kd_good += [k_d,]
            
    print "Sampling finished.\n\n"
    plt.plot(kp_unstable, kd_unstable, 'r o', label='unstable')
    plt.plot(kp_good,     kd_good,     'k o', label='good')
    kp = np.arange(0.0, MAX_KP, 1.0)
    kd = 2.0*np.sqrt(kp)
    plt.plot(kp, kd, label='kd=2*sqrt(kp)')
    plt.legend()
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.show()
    
    
if(TEST_CLOSED_LOOP_SEA>0):
    print "\n******************* TEST CLOSED-LOOP SEA *******************"
    print "Gonna perform %d tests to check that closed-loop SEA dynamics" % (TEST_CLOSED_LOOP_SEA)
    print "is correct."

    DO_PLOTS = 1
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
    
        cl_sea = CloseLoopSEA(dt, k, I_j, b_j, I_m, b_m, k_p, k_d, k_tau, b_tau)
        stableRH = cl_sea.check_RH_conditions()
        if(not stableRH):
            n_unstable += 1
            continue
    
        # initialize system state
        sea.x    = x_0
        cl_sea.x = x_0
    
        x    = np.zeros((N,4))
        x_cl = np.zeros((N,4))
        u = np.zeros(N)
        tau_d = np.zeros(N)
        x[0,:] = x_0
        x[0,2] = sea.tau()
        x[0,3] = sea.dtau()
        x_cl[0,:] = x_0
        x_cl[0,2] = cl_sea.tau()
        x_cl[0,3] = cl_sea.dtau()
        for t in range(N-1):
            tau_d[t] = -k_p*sea.qj() -k_d*sea.dqj()
            u[t] = -k_tau*(sea.tau() - tau_d[t]) - b_tau*sea.dtau()
            x[t+1,:] = sea.simulate(u[t])
            x[t+1,2] = sea.tau()
            x[t+1,3] = sea.dtau()
        
            x_cl[t+1,:] = cl_sea.simulate()
            x_cl[t+1,2] = cl_sea.tau()
            x_cl[t+1,3] = cl_sea.dtau()
        
        print "Max position trajectory error: ", np.max(np.abs(x[:,0]-x_cl[:,0]))
        
        if(DO_PLOTS):
            f, ax = plt.subplots(5,1,sharex=True);
            labels = ['qj','dqj','tau','dtau']
            for j in range(4):
                ax[j].plot(x[:,j],    '-',  label=labels[j]);
                ax[j].plot(x_cl[:,j], 'r--', label=labels[j]);
                if(j==2):
                    ax[j].plot(tau_d, label='tau_des')
                ax[j].legend()
            ax[-1].plot(u, label='tau_m')
            ax[-1].legend()
            plt.show();
    
    print "Test finished. Number of unstable gains found: %d" % (n_unstable)
