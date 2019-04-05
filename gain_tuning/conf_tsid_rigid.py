# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Configuration for tuning the gains for TSID-Rigid

@author: adelprete
"""

#!/usr/bin/env python
from numpy import matlib
from conf_common import *
    
GAINS_FILE_NAME         = 'gains_tsid_rigid'
controllers             = ['tsid_rigid']
ctrl_long_name          = 'TSID-Rigid'

ny          = 2     # size of configuration vector (i.e. com)
nf          = 4
x0          = matlib.zeros((4*ny,1))
x0[ny,0]    = .1     # initial CoM velocity in Y direction

w_d4x_list  = np.logspace(-1.5, -3.5, num=7)