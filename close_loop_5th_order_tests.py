import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
dt=0.001
          
def cost(gains, N=1000, plot=False):
    '''integral cost on the step responce'''
    N = 1000
    K0 = gains[0]
    K1 = gains[1]
    K2 = gains[2]
    T  = gains[3]
    P  = gains[4]
    x0 = np.matrix([.1,0,0,0,0]).T
    A = np.matrix([[0,    1,    0,    0,  0   ],
                   [0,    0,    1,    0,  0   ],
                   [0,    0,    0,    1,  0   ],
                   [K0,   K1,   K2,   0,  P   ],
                   [K0/T, K1/T, K2/T, 0, -1/T]])      
    x = x0.copy()
    err = []
    cost = 0
    for i in range(N):
        x = x+A*x*dt;
        if plot:
            err.append(x.A1[0])
        cost+=abs(x.A1[0])
    if plot:
        plt.plot(err)
        plt.show()
    return cost
    

import  scipy
#Initial gains
gains0 = np.array([0, 0, 0, 1.5, 0])

#optimize gain 
#TODO use an non convex fnc optimizer
from scipy.optimize import minimize
opt_gains = minimize(cost, gains0, method='L-BFGS-B').x

#Plot optimum
cost(opt_gains,N=1000,plot=True)

embed()
