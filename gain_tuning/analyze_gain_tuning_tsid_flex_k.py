# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:57:32 2018

Analyze the performance of TSID-Flex-K with different sets of gains.

@author: adelprete
"""

from numpy import matlib
import numpy as np

from simple_biped.utils.LDS_utils import compute_integrator_dynamics
import simple_biped.gain_tuning.conf_tsid_flex_k as conf
from simple_biped.gain_tuning.analyze_common import analyze_results

np.set_printoptions(precision=1, linewidth=200, suppress=True)

nc = conf.nc
(H, A, B) = compute_integrator_dynamics(matlib.zeros((nc,4*nc)))
P       = matlib.eye(5*nc)

def compute_system_matrices(gains):
    In = matlib.eye(conf.nc)
    K = np.hstack([gains[0]*In, gains[1]*In, gains[2]*In, gains[3]*In])
    return A, B, K

analyze_results(conf, compute_system_matrices, P)
