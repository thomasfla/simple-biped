# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:57:32 2018

Analyze the performance of TSID-Rigid with different sets of gains.

@author: adelprete
"""

from numpy import matlib
import numpy as np

from simple_biped.utils.LDS_utils import compute_integrator_dynamics

from simple_biped.gain_tuning.analyze_common import analyze_results

np.set_printoptions(precision=1, linewidth=200, suppress=True)

def analyze_gain_tuning_tsid_rigid(conf):
    nc = conf.nc
    (H, A, B) = compute_integrator_dynamics(matlib.zeros((nc,4*nc)))
    In = matlib.eye(conf.nc)
    A[nc:,:] = 0.0
    B[nc:2*nc,:] = In
    B[3*nc:,:] = 0.0
    P       = matlib.eye(5*nc)
    
    def compute_system_matrices(gains):    
        K = np.hstack([gains[0]*In, gains[1]*In, 0.0*In, 0.0*In])
        return A, B, K
    
    analyze_results(conf, compute_system_matrices, P)


if __name__=='__main__':
    import simple_biped.gain_tuning.conf_tsid_rigid as conf
    analyze_gain_tuning_tsid_rigid(conf)