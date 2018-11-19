# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 15:09:54 2018

@author: adelprete
"""

import numpy as np
from numpy import matlib

''' Compute the inequality constraints that ensure the contact force (expressed in world frame)
        is inside the (linearized) friction cones: B*x <= b
        @param mu Friction coefficient
        @param fMin Minimum normal force
    '''
def createContactForceInequalities(mu, fMin=0.0):
    if(fMin==0.0):
        k = 2
    else:
        k = 3
    B = matlib.zeros((k,2))
    b = matlib.zeros(k).T
    B[0,0]   = +1;   #  f_y - mu*f_z <= 0.0
    B[1,0]   = -1;   # -f_y - mu*f_z <= 0.0
    B[:,1]   = -mu;
    
    # minimum normal force
    if(k==3):
        B[-1,1] = -1;   # f_z >= f_min
        b[-1]   = -fMin;
    return (B,b);