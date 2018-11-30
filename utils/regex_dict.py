# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:35:34 2018

@author: adelprete
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import simple_biped.utils.plot_utils as plut

class RegexDict(dict):
    
    def generate_key(self, keys, values):
        res = ''
        for (k,v) in zip(keys, values):
            res += k+'='+str(v)+' '
        return res[:-1]
        
    def extract_key_value(self, multikey, key):
        for s in multikey.split():
            k,v = s.split('=')
            if k==key:
                return v
        
    def get_matching_keys(self, keys, values):
        matching_keys = self.keys()
        for (k,v) in zip(keys, values):
            if(v is not None):
                regexp = '.*'+k+'='+str(v)+'.*'
                matching_keys = [key for key in matching_keys if re.match(regexp, key)]
#            print "Keys matching %s:"%(regexp)
#            for key in matching_keys: print '   ', key
        return matching_keys
        
    def get_matching(self, keys, values):
        matching_keys = self.get_matching_keys(keys, values)
        return (self[key] for key in matching_keys)


def plot_from_multikey_dict(d, keys, x_var, y_var, fixed_params, variab_param, xscale=None, yscale=None):
    matching_keys = d.get_matching_keys(keys, fixed_params)
    x, y = {}, {}
    for mk in matching_keys:
        z_i = d.extract_key_value(mk, variab_param)
        if z_i not in x: x[z_i], y[z_i] = [], []    
        data = d[mk]
        x[z_i] += [float(d.extract_key_value(mk, x_var))]
        y[z_i] += [data.__dict__[y_var]]
    
    plt.figure()
    for key in x.keys():
        xy = np.array(zip(x[key], y[key]), dtype=[('x', float), ('y', float)])
        xy_sorted = np.sort(xy, order='x')
        x_sorted = [xi for (xi,yi) in xy_sorted]
        y_sorted = [yi for (xi,yi) in xy_sorted]
        plt.plot(x_sorted, y_sorted, '-*', label=key)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.legend()
    if xscale is not None: plt.xscale(xscale)
    if yscale is not None: plt.yscale(yscale)
    fig_name = y_var+'_VS_'+x_var
    for (k,v) in zip(keys, fixed_params):
        if(v is not None):
            fig_name += '__'+k+'='+str(v)
    plut.saveFigure(fig_name)
    return (x,y)

if __name__ == '__main__':
    keys = ['ctrl', 'zeta', 'k']
    key_values = [['adm', 0.3, 1.0], 
                  ['adm', 0.1, 1.0], 
                  ['tsid_adm', 0.3, 1.0], 
                  ['tsid_adm', 0.3, 2.0]]
    rd = RegexDict()
    for kv in key_values:
        key = rd.generate_key(keys, kv)
        rd[key] = key

    patterns = [['adm', None, None],
                ['tsid_adm', None, None],
                [None, 0.3, None],
                [None, 0.1, None],
                [None, 0.3, 1.0]]
    for p in patterns:
        print 'Elements matching pattern', p
        for o in rd.get_matching(keys, p):
            print o
        print "".center(100,'#')
