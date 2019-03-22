# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Configuration for tuning the gains of all controllers

@author: adelprete
"""

#!/usr/bin/env python
from __future__ import print_function
from shutil import copyfile

from simple_biped.gain_tuning.analyze_common import compare_controllers

import simple_biped.gain_tuning.conf_adm_ctrl as conf_adm_ctrl
import simple_biped.gain_tuning.conf_tsid_adm as conf_tsid_adm
import simple_biped.gain_tuning.conf_tsid_flex_k as conf_tsid_flex_k
import simple_biped.gain_tuning.conf_tsid_rigid as conf_tsid_rigid

TESTS_DIR_NAMES = ['push_recovery_slip',
                   'push_recovery_ekf_slip',
                   'push_recovery_ekf_slip_coulomb',
                   'push_recovery_ekf_slip_coulomb_bw',
                   'push_recovery_ekf_slip_bw',
                   'push_recovery_slip_bw']
    
conf_list = [conf_tsid_rigid, conf_adm_ctrl, conf_tsid_adm, conf_tsid_flex_k]
marker_list = ['s', 'o', '*', 'v']

DATA_DIR = conf_list[0].DATA_DIR
for (i,name) in enumerate(TESTS_DIR_NAMES):
    print(name)
    for conf in conf_list:
        conf.TESTS_DIR_NAME = name+'/'
    compare_controllers(conf_list, marker_list)
    
    copyfile(DATA_DIR+name+'/roc_performance_comparison.png', DATA_DIR+'%2d'%(i)+'_'+name+'.png')