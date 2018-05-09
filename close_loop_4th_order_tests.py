import numpy as np
import matplotlib.pyplot as plt
#from IPython import embed
#from scipy.optimize import minimize
from scipy.optimize import basinhopping
from numpy.linalg import eigvals
from scipy.linalg import expm

np.set_printoptions(precision=2, suppress=True, linewidth=100);

print "\n**************************** CLOSE LOOP 4TH OTDER TEST ****************************"
print "Simple test to check if it is possible to stabilize a linear 4-th order system with"
print "a zero feedback gain on the 3rd derivative of the position variable."

def compute_A_matrix(gains, T_min=1e-5):
    K0 = gains[0]
    K1 = gains[1]
    K2 = gains[2]
    A = np.array( [[0,         1,        0,       0],
                   [0,         0,        1,       0],
                   [0,         0,        0,       1],
                   [-K0,      -K1,      -K2,      0]])
    return A;
           
def max_eigen_value(gains):
    A = compute_A_matrix(gains);
    ei = eigvals(A);
    return np.max(np.real(ei))
    
def step_response(gains, dt=0.001, N=3000, plot=False):
    '''integral cost on the step responce'''
    n = 4;
    A = compute_A_matrix(gains);
    x0 = np.array([.1,0,0,0]).T
    x = np.empty((N,n))
    x[0,:] = x0
    cost = 0.0
    e_dtA = expm(dt*A);
#    e_dtA = np.identity(5) + dt*A;
    for i in range(N-1):
        x[i+1,:] = np.dot(e_dtA, x[i,:]);
        cost += abs(x[i+1,0])
        
    if plot:
        f, ax = plt.subplots(n,1,sharex=True);
        for i in range(n):
            ax[i].plot(x[:,i])
        plt.show()
#    print cost, gains;
    return cost
    
def callback(x, f, accept):
    global nit
    print "%4d) Cost: %10.3f; Accept %d Gains:"%(nit, f, accept), x
    nit += 1;
    

#Initial gains
gains0 = np.array([1.0, 0.0, 0.0]);
gains0 = np.array([15076.06,     -0.  ,  49032.56]);

#Plot step response
print "Initial gains:", gains0
A = compute_A_matrix(gains0);
print "Initial eigenvalues:\n", eigvals(A).T;
step_response(gains0,N=10000,plot=True)


#optimize gain 
nit = 0;
opt_res = basinhopping(max_eigen_value, gains0, niter=10000, disp=False, T=0.1, stepsize=100.0, callback=callback)
opt_gains = opt_res.x;
print opt_res;

#Plot step response
A = compute_A_matrix(opt_gains);
print "Optimal gains:      ", opt_gains
print "Optimal eigenvalues:", eigvals(A);
step_response(opt_gains,N=10000,plot=True)
