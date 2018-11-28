# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

@author: adelprete
"""

#!/usr/bin/env python
from __future__ import print_function
import itertools
from select     import select
from subprocess import Popen, PIPE, STDOUT
import time

controllers = ['tsid_rigid', 'tsid_flex', 'tsid_adm', 'tsid_mistry', 'adm_ctrl']
f_dists = [400.]
com_sin_amps = [0.0]
zetas = [0.1, 0.2, 0.4, 0.8]
ks = [0.1, 0.2, 0.4, 1.0, 2.0]
T = 2.0

processes = []
for (ctrl, f_dist, com_sin_amp, zeta, k) in itertools.product(controllers, f_dists, com_sin_amps, zetas, ks):
    cmd = ('python track_com.py --controller=%s --com_sin_amp=%.2f --f_dist=%.1f --zeta=%.1f --T=%.1f --k=%.2f'%(ctrl, com_sin_amp, f_dist, zeta, T, k))
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