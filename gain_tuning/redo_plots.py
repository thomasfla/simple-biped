# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Configuration for tuning the gains of all controllers

@author: adelprete
"""

#!/usr/bin/env python
from __future__ import print_function
from shutil import copyfile
import matplotlib.pyplot as plt

from simple_biped.gain_tuning.analyze_common import compare_controllers

from simple_biped.gain_tuning.analyze_gain_tuning_tsid_rigid import analyze_gain_tuning_tsid_rigid
from simple_biped.gain_tuning.analyze_gain_tuning_tsid_flex_k import analyze_gain_tuning_tsid_flex_k
from simple_biped.gain_tuning.analyze_gain_tuning_adm_ctrl import analyze_gain_tuning_adm_ctrl
from simple_biped.gain_tuning.analyze_gain_tuning_tsid_adm import analyze_gain_tuning_tsid_adm

import simple_biped.gain_tuning.conf_adm_ctrl as conf_adm_ctrl
import simple_biped.gain_tuning.conf_tsid_adm as conf_tsid_adm
import simple_biped.gain_tuning.conf_tsid_flex_k as conf_tsid_flex_k
import simple_biped.gain_tuning.conf_tsid_rigid as conf_tsid_rigid

REDO_ANALYSIS = 0
TESTS_DIR_NAMES = ['push_recovery_slip',
                   'push_recovery_ekf_slip',
                   'push_recovery_ekf_slip_coulomb',
                   'push_recovery_ekf_slip_coulomb_bw',
                   'push_recovery_ekf_slip_bw',
                   'push_recovery_slip_bw']

#TESTS_DIR_NAMES = ['push_recovery_ekf_slip'] #, 'push_recovery_ekf_slip']
    
conf_list = [conf_tsid_rigid, conf_adm_ctrl, conf_tsid_adm, conf_tsid_flex_k]
marker_list = ['s', 'o', '*', 'v']
analysis_functions = [analyze_gain_tuning_tsid_rigid, 
                      analyze_gain_tuning_adm_ctrl, 
                      analyze_gain_tuning_tsid_adm,
                      analyze_gain_tuning_tsid_flex_k]

DATA_DIR = conf_list[0].DATA_DIR
for (i,name) in enumerate(TESTS_DIR_NAMES):
    print(i, name)
    for (conf, analyze) in zip(conf_list, analysis_functions):
        conf.TESTS_DIR_NAME = name+'/'
        if(REDO_ANALYSIS):
            analyze(conf)
        
    compare_controllers(conf_list, marker_list)
    plt.close('all')
    
    index = i
    copyfile(DATA_DIR+name+'/roc_performance_comparison.png', DATA_DIR+'%2d'%(index)+'_'+name+'.png')
    copyfile(DATA_DIR+name+'/roc_performance_comparison_jerk.png', DATA_DIR+'jerk_%2d'%(index)+'_'+name+'.png')
    copyfile(DATA_DIR+name+'/roc_performance_comparison_jerk_max.png', DATA_DIR+'max_jerk_%2d'%(index)+'_'+name+'.png')