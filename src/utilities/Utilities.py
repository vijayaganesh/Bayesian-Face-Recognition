# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 17:57:13 2018

@author: VijayaGanesh Mohan
"""
import math
import numpy as np
class Utilities:
    @staticmethod
    def line_search_min(func,mn,mx,steps = 10000):
        out_dict = {}
        for i in np.linspace(mn,mx+1,steps):
            a = func(i)
            out_dict[a] = i
        min_val = min(out_dict.keys())
        return out_dict[min_val]     
    @staticmethod
    def pdf_mnd(mu,cov,x):
        x = np.array([x])
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        prod = -0.5 * np.matmul(np.matmul(x-mu,cov_inv),np.transpose(x-mu))
        prob = (1/(2*np.pi*cov_det)**0.5)*np.exp(prod)
        return prob
    
if __name__ == '__main__':
    print('Local Minimum: '+repr(Utilities.line_search_min(math.sin,1,2*math.pi)))

    