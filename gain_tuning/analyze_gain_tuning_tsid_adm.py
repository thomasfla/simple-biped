# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:57:32 2018

Analyze the performance of TSID-Admittance with different sets of gains.

@author: adelprete
"""

from numpy import matlib
import numpy as np

from simple_biped.utils.LDS_utils import compute_integrator_dynamics
from simple_biped.tsid_admittance import GainsTsidAdm
from simple_biped.gain_tuning.tune_gains_tsid_adm_utils import convert_tsid_adm_gains_to_integrator_gains
from simple_biped.simu import Simu
from simple_biped.gain_tuning.analyze_common import analyze_results

np.set_printoptions(precision=1, linewidth=200, suppress=True)

def analyze_gain_tuning_tsid_adm(conf):
    K_contact = Simu.get_default_contact_stiffness()
    nc = conf.nc
    (H, A, B) = compute_integrator_dynamics(matlib.zeros((nc,4*nc)))
    P       = matlib.eye(5*conf.nc)
    
    def compute_system_matrices(gains):
        gains_TSID_adm = GainsTsidAdm(gains)
        K = convert_tsid_adm_gains_to_integrator_gains(gains_TSID_adm, K_contact, conf.nc)
        return A, B, K
    
    analyze_results(conf, compute_system_matrices, P)

if __name__=='__main__':
    import simple_biped.gain_tuning.conf_tsid_adm as conf
    analyze_gain_tuning_tsid_adm(conf)