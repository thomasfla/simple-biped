# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 14:35:34 2018

@author: adelprete
"""

import re

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
#        matching_keys = self.keys()
#        for (k,v) in zip(keys, values):
#            if(v is not None):
#                regexp = '.*'+k+'='+str(v)+'.*'
#                matching_keys = [key for key in matching_keys if re.match(regexp, key)]
#            print "Keys matching %s:"%(regexp)
#            for key in matching_keys: print '   ', key
        matching_keys = self.get_matching_keys(keys, values)
        return (self[key] for key in matching_keys)


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
