# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 09:54:42 2018

Run several tests with TSID Admittance with different gains.

@author: adelprete
"""

#!/usr/bin/env python
import simple_biped.gain_tuning.conf_tsid_adm as conf
from simple_biped.gain_tuning.run_tests_common import run_tests

run_tests(conf)