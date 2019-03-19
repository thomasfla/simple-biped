# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:57:32 2018

Analyze the performance of a controller with different sets of gains.

@author: adelprete
"""

from numpy import matlib
import numpy as np
from simple_biped.gain_tuning.tune_gains_adm_ctrl_utils import GainOptimizeAdmCtrl, compute_projection_to_com_state_ctrl
from simple_biped.simu import Simu
from simple_biped.admittance_ctrl import GainsAdmCtrl
from simple_biped.hrp2_reduced import Hrp2Reduced
from simple_biped.robot_model_path import pkg, urdf 
import simple_biped.gain_tuning.conf_adm_ctrl as conf
from simple_biped.gain_tuning.analyze_common import analyze_results

# SETUP
robot   = Hrp2Reduced(urdf, [pkg], loadModel=0, useViewer=0)
q       = robot.q0.copy()
v       = matlib.zeros((robot.model.nv,1))
ny, nf, dt = conf.ny, conf.nf, conf.dt_simu
N       =  int(conf.T_simu/dt)

K               = Simu.get_default_contact_stiffness()
initial_gains   = GainsAdmCtrl.get_default_gains(K).to_array()
gain_optimizer  = GainOptimizeAdmCtrl(robot, q, v, K, ny, nf, initial_gains, dt, conf.x0, N, np.eye(2*ny+3*nf))

P = compute_projection_to_com_state_ctrl()

analyze_results(conf, gain_optimizer.compute_system_matrices, P)
