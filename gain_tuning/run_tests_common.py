# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Run several tests with different gains.

@author: adelprete
"""

#!/usr/bin/env python
from __future__ import print_function
import itertools
from select     import select
from subprocess import Popen, PIPE, STDOUT
import time

def run_tests(conf):
    controllers     = conf.controllers 
    f_dists         = conf.f_dists
    zetas           = conf.zetas
    T               = conf.T_simu
    w_d4x_list      = conf.w_d4x_list
    
    processes = []
    for (ctrl, f_dist, zeta, w_d4x) in itertools.product(controllers, f_dists, zetas, w_d4x_list):
        test_name = conf.TESTS_DIR_NAME + conf.controllers[0]+'/'+conf.get_test_name(ctrl, zeta, f_dist, w_d4x)
        gains_file = conf.DATA_DIR + conf.GAINS_DIR_NAME + conf.get_gains_file_name(conf.GAINS_FILE_NAME, w_d4x)
        
        cmd = ('python ../track_com.py --controller=%s --f_dist=%.1f --zeta=%.1f --T=%.1f --gain_file="%s" --test_name="%s"'%(
                                            ctrl, f_dist, zeta, T, gains_file, test_name))
        print("Execute this command:\n", cmd, "\n")
        # start several subprocesses
        processes += [Popen(cmd.split(), stdout=PIPE, stderr=STDOUT, bufsize=1, close_fds=True, universal_newlines=True)]
        time.sleep(conf.TIME_BETWEEN_TESTS)
        
    # read output
    timeout = 1.0 # seconds
    while processes:
        # remove finished processes from the list (O(N**2))
        for p in processes[:]:
            if p.poll() is not None: # process ended
                print(p.stdout.read(), end='') # read the rest
                p.stdout.close()
                processes.remove(p)
                print('\n', (" A process has terminated. %d processes still running "%len(processes)).center(100,'#'), '\n')
    
        # wait until there is something to read
        rlist = select([p.stdout for p in processes], [],[], timeout)[0]
    
        # read a line from each process that has output ready
        for f in rlist:
            print(f.readline(), end='') #NOTE: it can block
