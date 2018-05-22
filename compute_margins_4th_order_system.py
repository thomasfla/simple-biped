import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
from numpy.random import uniform
from control import tf, margin, bode_plot, step_response, feedback, nyquist_plot

np.set_printoptions(precision=2, suppress=True, linewidth=100);

print "\n******************* COMPUTE PHASE-GAIN MARGINS FOR 4-TH ORDER SYSTEM  *******************"
print "Compute margins for stability of linear 4-th order system"

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

def check_RH_conditions(gains):
    kp = gains[3]
    kd = gains[2]
    ka = gains[1]
    kj = gains[0]
    if(kp<=0.0 or kd<=0.0 or ka<=0.0 or kj<=0.0):
        return False;
    if(ka*kj <= kd):
        return False;
    if(kd*ka*kj - kd*kd <= kp*kj*kj):
        return False
    return True
    
#Initial gains
MAX_GAIN = 1e2
N_TESTS = int(1e3)
kj = 2
ka = 10
print "Gonna perform %d tests with random gains between 0 and %.1f\n" % (N_TESTS, MAX_GAIN)

for i in range(N_TESTS):
    gains = uniform(low=4*(0.0,), high=4*(MAX_GAIN,))
    gains[0] = kj;
    gains[1] = ka;

    stableRH = check_RH_conditions(gains)
    if(not stableRH):
        continue;
        
    maxEv = max_eigen_value(gains)
        
    ol_sys = tf(-gains, np.array([1.0, 0, 0, 0, 0]))
    gm, pm, Wcg, Wcp = margin(ol_sys)
    if(gm is None):
        gm = np.inf;
    print "Stable %d\t Max(Ev)=%4.1f \t Margins: %5.2f \t %5.2f" % (stableRH, maxEv, gm, pm);
    
#    bode_plot(ol_sys);
#    nyquist_plot(ol_sys)
#    print "cl sys:", cl_sys
    cl_sys = feedback(ol_sys, sign=1);
    T, yout = step_response(cl_sys)
    plt.plot(T, yout);
    plt.show();
    
    
   
print "\nTests finished"
