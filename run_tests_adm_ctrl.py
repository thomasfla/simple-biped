# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Run several tests with admittance control with different gains.

@author: adelprete
"""

#!/usr/bin/env python
from __future__ import print_function
import itertools
from select     import select
from subprocess import Popen, PIPE, STDOUT
import time
import os
import numpy as np

GAINS_DIR         = os.getcwd()+'/data/gains/'
GAINS_FILE_NAME   = 'gains_adm_ctrl'
controllers = ['adm_ctrl']
f_dists = [400.]
zetas = [0.3]
T = 3.
w_ddf_list  = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6] #[1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]

processes = []
for (ctrl, f_dist, zeta, w_ddf) in itertools.product(controllers, f_dists, zetas, w_ddf_list):
    test_name = 'test_gain_tuning/'+ctrl + '_zeta_'+str(zeta)+'_fDist_'+str(f_dist)+'_w_ddf_'+str(w_ddf)
    gains_file = GAINS_DIR + GAINS_FILE_NAME+'_w_ddf='+str(w_ddf)+'.npy'
    cmd = ('python track_com.py --controller=%s --f_dist=%.1f --zeta=%.1f --T=%.1f --gain_file="%s" --test_name="%s"'%(
                                        ctrl, f_dist, zeta, T, gains_file, test_name))
    print("Execute this command:\n", cmd, "\n")
    # start several subprocesses
    processes += [Popen(cmd.split(), stdout=PIPE, stderr=STDOUT, bufsize=1, close_fds=True, universal_newlines=True)]
    time.sleep(1.)
    
# read output
timeout = 0.1 # seconds
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