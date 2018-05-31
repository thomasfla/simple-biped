import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
from numpy.random import uniform
from scipy.linalg import expm
from math import sqrt
#from control import tf, margin, bode_plot, step_response, feedback, nyquist_plot

np.set_printoptions(precision=2, suppress=True, linewidth=100);

print "\n******************* COMPUTE PERFORMANCE FOR 4-TH ORDER SYSTEM  *******************"
print "Compute performance (rise-time, overshoot, settling time) for LTI 4-th order system"
print "as a function of PD CoM gains"

def compute_A_matrix(gains):
    kp = gains[3]
    kd = gains[2]
    ka = gains[1]
    kj = gains[0]
    A = np.array( [[0,         1,        0,       0],
                   [0,         0,        1,       0],
                   [0,         0,        0,       1],
                   [-kp,      -kd,     -ka,     -kj]])
    return A;
           
def max_eigen_value(gains):
    A = compute_A_matrix(gains);
    ei = eigvals(A);
    return np.max(np.real(ei))
    
def step_response(gains, dt, N):
    '''Compute step response'''
    n = 4;
    A = compute_A_matrix(gains);
    x = np.zeros((N,n));
    x[0,0] = 1.0;
    e_dtA = expm(dt*A);
    for i in range(N-1):
        x[i+1,:] = np.dot(e_dtA, x[i,:]);
        
    return x

def compute_4th_order_gains_from_admittance_gains(k, kf, kp, kd, kcp, kcd):
    k1 = k*kp*kf*kcp;
    k2 = k*kp*kf*kcd;
    k3 = kp*(1+k*kf);
    return np.array([kd, k3, k2, k1]);
    
# USER PARAMETERS
DO_PLOTS = 0
MAX_GAIN = [5e2, 1e2]
N_TESTS = int(1e4)
N = 6000               # number of time steps in step response
dt = 0.001              # sampling time of step response
FALL_PERCENTAGE = 0.1
SETTLING_PERCENTAGE = 0.05

FALL_TIME_THR = 1.0     # maximum desired fall time
SETT_TIME_THR = 2.0     # maximum desired settling time
OVERSHOOT_THR = 0.2     # maximum desired overshoot

k = 1e4         # contact stiffness
kp = 400        # admittance control proportional gain
kd = 2*sqrt(kp) # admittance control derivative gain
kf = 1.0/k      # force proportional gain

# for a proportional gain kp, the natural frequency for the 2-nd order system is w=sqrt(kp)
# Setting the damping kd=2*sqrt(kp), then the system response is:
#   x(t) = x(0)*(1+w*t)*e^(-w*t)
# If kp=400 => w=20 => x(0.5)=0.0005*x(0), x(0.2) = 0.09*x(0)
# If kp=100 => w=10 => x(1) = 0.0005*x(0), x(0.5)=0.04*x(0)
# If kp=25  => w=5  => x(1) = 0.04*x(0),   x(0.5)=0.3*x(0)

print "Gonna perform %d tests with random CoM gains: max kp=%.1f, max kd=%.1f\n" % (N_TESTS, MAX_GAIN[0], MAX_GAIN[1])

n_unstable = 0;
n_good_perf = 0;
kcp_unstable = [];
kcd_unstable = [];
kcp_bad = []
kcd_bad = []
kcp_good = []
kcd_good = []
for i in range(N_TESTS):
    kcp = uniform(low=0.0, high=MAX_GAIN[0])   # com proportional gain
    kcd = uniform(low=0.0, high=MAX_GAIN[1])   # com derivative gain
    
    gains = compute_4th_order_gains_from_admittance_gains(k, kf, kp, kd, kcp, kcd)
    
    maxEv = max_eigen_value(gains)
    if(maxEv>=0.0):
        n_unstable += 1;
        kcp_unstable += [kcp,]
        kcd_unstable += [kcd,]
        continue;
        
    x = step_response(gains, dt, N)
    
    overshoot = -np.min(x[:,0])

    if(-overshoot<FALL_PERCENTAGE):
        T_fall = np.where(x[:,0]<FALL_PERCENTAGE)[0][0] * dt
    else:
        T_fall = N*dt
        
    T_sett = np.where(np.abs(x[:,0])>SETTLING_PERCENTAGE)[0][-1] * dt
    
    if(overshoot<OVERSHOOT_THR and T_fall<FALL_TIME_THR and T_sett<SETT_TIME_THR):
        n_good_perf += 1
        kcp_good += [kcp,]
        kcd_good += [kcd,]
    else:
        kcp_bad += [kcp,]
        kcd_bad += [kcd,]
        
    #print "Max(Ev)=%4.1f \t Overshoot: %4.2f\t Fall time: %5.1f\t Settling time %5.1f" % (maxEv, overshoot, T_fall, T_sett)
    
    if(DO_PLOTS):
        plt.plot(x[:,0]);
        plt.plot([0, T_fall/dt],   [FALL_PERCENTAGE, FALL_PERCENTAGE], 'r:')
        plt.plot([T_sett/dt, N-1], [SETTLING_PERCENTAGE, SETTLING_PERCENTAGE], 'b:')
        plt.plot([T_sett/dt, N-1], [-SETTLING_PERCENTAGE, -SETTLING_PERCENTAGE], 'b:')
        plt.plot([0, np.argmin(x[:,0])], [-overshoot, -overshoot], 'g:')
        plt.show();

print "\n%d tests finished"%(N_TESTS)
print "Number of unstable tests:         ", n_unstable
print "Number of bad performance tests:  ", (N_TESTS-n_unstable-n_good_perf)
print "Number of good performance tests: ", n_good_perf

plt.plot(kcp_unstable, kcd_unstable, 'r o', label='unstable')
plt.plot(kcp_bad,      kcd_bad,      'b x', label='bad')
plt.plot(kcp_good,     kcd_good,     'k o', label='good')
kp = np.arange(0.0, MAX_GAIN[0], 1.0)
kd = 2.0*np.sqrt(kp)
plt.plot(kp, kd, label='kd=2*sqrt(kp)')
plt.legend()
plt.xlabel('Kp')
plt.ylabel('Kd')
plt.show()