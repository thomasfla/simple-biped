import numpy as np
import matplotlib.pyplot as plt
#from IPython import embed
#from scipy.optimize import minimize
from scipy.optimize import basinhopping
from numpy.linalg import eigvals
from scipy.linalg import expm

np.set_printoptions(precision=2, suppress=True, linewidth=100);

DISCRETE_TIME = 1

def compute_A_matrix(gains, dt=0.001, T_min=1e-5):
    K0 = gains[0]
    K1 = gains[1]
    K2 = gains[2]
    T  = gains[3]*1e-4
    P  = gains[4]
    if(T<=T_min):
        T = T_min;
    Tinv = 1.0/T;
    A = np.array( [[0,         1,        0,       0,   0   ],
                   [0,         0,        1,       0,   0   ],
                   [0,         0,        0,       1,   0   ],
                   [-K0,      -K1,      -K2,      0,  -P   ],
                   [-K0*Tinv, -K1*Tinv, -K2*Tinv, 0,  -Tinv]])
    if(DISCRETE_TIME):
        return np.identity(5) + dt*A
    return A;
           
def max_eigen_value(gains):
    A = compute_A_matrix(gains);
    ei = eigvals(A);
    if(DISCRETE_TIME):
        return np.max(np.abs(ei))
    return np.max(np.real(ei))
    
def step_response(gains, dt=0.001, N=3000):
    ''' compute step response '''
    A = compute_A_matrix(gains);
    n = A.shape[0];    
    x = np.zeros((N,n))
    x[0,0] = 1.0;

    if(DISCRETE_TIME):
        e_dtA = A;
    else:
        e_dtA = expm(dt*A);

    for i in range(N-1):
        x[i+1,:] = np.dot(e_dtA, x[i,:]);
        
    return x;
        
def plot_state_traj(x, title="", show_plot=True):
    n = x.shape[1];
    f, ax = plt.subplots(n,1,sharex=True);
    for i in range(n):
        ax[i].plot(x[:,i])
        ax[i].grid(True);
    ax[0].set_title(title)
    if(show_plot):
        plt.show()
    
def callback(x, f, accept):
    global nit
    if(accept):
        print "%4d) Cost: %10.3f; Gains:"%(nit, f), x
    nit += 1;
    

#Initial gains
if(DISCRETE_TIME):
    gains0 = np.array([0, 0, 0, 1.5*1e4, 0])
    gains0 = np.array([181.88254841,  78.38880159,  46.67521938,  620.42555387, 836.17375548])
    gains0 = np.array([2981.12,   696.78,   647.64,  3001.35,     0.84])
    gains0 = np.array([4998.82,   380.21,   235.61,  2354.81,     0.81]) # 0.999
    gains0 = np.array([8662.94,  1206.55,   411.52,  1189.86,     0.79]) # 0.998
    gains0 = np.array([8901.53,  1579.62,   382.2 ,   374.33,     0.85]) # 0.995
    gains0 = np.array([8768.24,  1894.39,   420.49,   344.12,     0.88]) # 0.994
    gains0 = np.array([8528.17,  2405.21,   521.33,   283.13,     0.91]) # 0.993
    gains0 = np.array([8504.81,  2475.27,   535.02,   277.36,     0.92]) # 0.992
else:
    gains0 = np.array([0, 0, 0, 1.5, 0])
    gains0 = np.array([181.88254841,  78.38880159,  46.67521938,  620.42555387, 836.17375548])
    gains0 = np.array([ 16400125.34,   9414488.71,   1374699.5 ,  1e-1,         0.99])

#Plot step response
print "Initial gains:", gains0
A = compute_A_matrix(gains0);
print "Initial eigenvalues:\n", eigvals(A).T;
x_gains0 = step_response(gains0,N=3000)
plot_state_traj(x_gains0);


#optimize gain 
nit = 0;

# a local minimum that is T/10 bigger than the current optimum is accepted with a probability of 90%
# a local minimum that is T/2  bigger than the current optimum is accepted with a probability of 60%
# a local minimum that is   T  bigger than the current optimum is accepted with a probability of 37%
# a local minimum that is 2*T  bigger than the current optimum is accepted with a probability of 13%
# a local minimum that is 5*T bigger than the current optimum is accepted with a probability of 0.01%
T = 0.003

opt_res = basinhopping(max_eigen_value, gains0, niter=10000, disp=False, T=T, stepsize=10.0, callback=callback)
opt_gains = opt_res.x;
print opt_res;

#Plot step response
A = compute_A_matrix(opt_gains);
ei = eigvals(A);
print "Optimal gains:      ", opt_gains
print "Optimal eigenvalues:", ei;
if(DISCRETE_TIME):
    print "Eigenvalue norms (*1e3):", 1e3*np.abs(ei);
x_opt_gains = step_response(opt_gains,N=3000)

plot_state_traj(x_gains0, "Initial gains", False);
plot_state_traj(x_opt_gains, "Optimal gains");

print "\n\ngains0 = np.array([", opt_gains, "])";