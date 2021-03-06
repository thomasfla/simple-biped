# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Configuration for tuning the gains for TSID-Admittance

@author: adelprete
"""

#!/usr/bin/env python
from numpy import matlib
from conf_common import *
    
#TESTS_DIR_NAME          = 'test_gain_tuning/tsid_amd_w_dx_0_d2x_0_d3x_0/'
GAINS_FILE_NAME         = 'gains_tsid_adm'
controllers             = ['tsid_adm']
ctrl_long_name          = 'TSID-Adm'

ny          = 2     # size of configuration vector (i.e. com)
nf          = 4
x0          = matlib.zeros((4*ny,1))
x0[ny,0]    = .1     # initial CoM velocity in Y direction
