# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:57:32 2018

Generate plots starting from data logger npz files.

@author: adelprete
"""

from numpy import matlib
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from simple_biped.utils.logger import RaiLogger
from simple_biped.hrp2_reduced import Hrp2Reduced
from simple_biped.robot_model_path import pkg, urdf 
import simple_biped.utils.plot_utils as plut
from simple_biped.utils.plot_utils import plot_from_logger

import os

class Empty:
    pass    
    
np.set_printoptions(precision=1, linewidth=200, suppress=True)


# User parameters
DATA_DIR                = str(os.path.dirname(os.path.abspath(__file__)))+'/../data/data_comparison_paper_v3/'
dt                      = 1e-3
mu_simu = 0.3

DATA_FILE_NAME          = 'logger_data.npz'
plut.SAVE_FIGURES       = 1
SHOW_FIGURES            = 1
plut.FIGURE_PATH        = DATA_DIR

# SETUP
robot   = Hrp2Reduced(urdf, [pkg], loadModel=0, useViewer=0)    
#time = np.arange(N*dt, step=dt)

print 'Gonna analyze %d folders found in folder %s'%(len(os.listdir(DATA_DIR)), DATA_DIR);
for dirname in os.listdir(DATA_DIR):                
    path = DATA_DIR+dirname+'/';
    if(not os.path.exists(path)):
        print "ERROR Path ", path, "does not exist!";
        continue;
                
    INPUT_FILE = path + DATA_FILE_NAME
    
    print '\n'+"".center(120, '#')
    print ("Gonna read %s"%(dirname)).center(120)
    print "".center(120, '#')
        
    # SETUP LOGGER
    lgr = RaiLogger()
    try:
        lgr.load(INPUT_FILE)
    except:
        print "Could not read file", INPUT_FILE
        continue
 
    fields, labels, linest = [], [], []
    fields += [['simu_lkf_0',          'tsid_lkf_0',     'simu_rkf_0',          'tsid_rkf_0']]
    labels += [['left',                'left des',       'right',               'right des']]
    linest += [['b', '--', 'r', '--']]
    fields += [['simu_lkf_1',          'tsid_lkf_1',     'simu_rkf_1',          'tsid_rkf_1']]
    labels += [['left',                'left des',       'right',               'right des']]
    linest += [['b', '--', 'r', '--']]
    ax = plot_from_logger(lgr, dt, fields, labels, 'Contact Forces', linest, ylabel=['Y [N]', 'Z [N]'])
    plut.FIGURE_PATH        = path
    plut.saveFigure('contact_forces')
    
    N = len(lgr.get_streams('simu_q_0'))
    tt = np.arange(0.0, dt*N, dt)
    simu_lfk_z = np.array(lgr.get_streams('simu_lkf_1'))
    simu_rfk_z = np.array(lgr.get_streams('simu_rkf_1'))
    ax[0].plot(tt,  mu_simu*simu_lfk_z, 'b:', label='left bounds')
    ax[0].plot(tt, -mu_simu*simu_lfk_z, 'b:') #, label='left min')
    ax[0].plot(tt,  mu_simu*simu_rfk_z, 'r:', label='right bounds')
    ax[0].plot(tt, -mu_simu*simu_rfk_z, 'r:') #, label='right min')
    ax[0].legend()
    plut.saveFigure('contact_forces_with_friction_bounds')