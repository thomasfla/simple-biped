# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Configuration for tuning the gains for TSID-Admittance

@author: adelprete
"""

#!/usr/bin/env python
from __future__ import print_function

import os
import numpy as np
from numpy import matlib

from conf_common import *
    
def get_gains_file_name(w_d4x):
    return GAINS_FILE_NAME+'_w_d4x='+str(w_d4x)+'.npy'
    
TESTS_DIR_NAME          = 'test_gain_tuning/tsid_amd_w_dx_0_d2x_0_d3x_0/'
GAINS_FILE_NAME         = 'gains_tsid_adm'
controllers             = ['tsid_adm']

ny          = 1
nf          = 4
x0          = matlib.zeros((4*ny,1))
x0[1,0]    = .1     # initial CoM velocity in Y direction

Q_pos       = matlib.diagflat(np.matrix(ny*[w_x] + ny*[w_dx] + ny*[w_d2x] + ny*[w_d3x]))
Q_d4x       = matlib.diagflat(np.matrix(ny*[0.0] + ny*[0.] + ny*[0.] + ny*[0.]))
R_pos       = matlib.diagflat(np.matrix(ny*[0.0]))
R_d4x       = matlib.diagflat(np.matrix(ny*[1.0]))