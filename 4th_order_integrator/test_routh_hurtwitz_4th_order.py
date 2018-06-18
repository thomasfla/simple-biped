import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals
from numpy.random import uniform

np.set_printoptions(precision=2, suppress=True, linewidth=100);

print "\n******************* TEST ROUTH-HURTWITZ 4-TH ORDER TEST *******************"
print "Simple test to check Routh-Hurtwitz condition for stability of linear 4-th order system"

def compute_A_matrix(gains):
    kp = gains[0]
    kd = gains[1]
    ka = gains[2]
    kj = gains[3]
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
    kp = gains[0]
    kd = gains[1]
    ka = gains[2]
    kj = gains[3]
    if(kp<=0.0 or kd<=0.0 or ka<=0.0 or kj<=0.0):
        return False;
    if(ka*kj <= kd):
        return False;
    if(kd*ka*kj - kd*kd <= kp*kj*kj):
        return False
    return True
    
#Initial gains
MAX_GAIN = 1e5
N_TESTS = int(1e7)

print "Gonna perform %d tests with random gains between 0 and %.1f\n" % (N_TESTS, MAX_GAIN)

n_err = 0;
n_stable = 0;
for i in range(N_TESTS):
    gains = uniform(low=4*(0.0,), high=4*(MAX_GAIN,))
    maxEv = max_eigen_value(gains)
    stableRH = check_RH_conditions(gains)
    
    if(stableRH):
        n_stable += 1;
        if(maxEv>=0.0):
            n_err += 1;
            print "[ERROR] gains:", gains.T
            print "        Routh-Hurtwitz conditions say system should be stable, but max eigenvalue is", maxEv
    elif(maxEv<0.0):
        n_err += 1;
        print "[ERROR] gains:", gains.T
        print "        Routh-Hurtwitz conditions say system should be unstable, but max eigenvalue is", maxEv
   
print "Test finished. Number of stable gains found: %d" % (n_stable)
print "\n%d errors out of %d tests" %(n_err, N_TESTS)
