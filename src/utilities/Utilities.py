# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 17:57:13 2018

@author: VijayaGanesh Mohan
"""
import math
import numpy as np
from scipy.special import gammaln
class Utilities:
    @staticmethod
    def line_search_min(func,mn,mx,steps = 10000,*args):
        out_dict = {}
        for i in np.linspace(mn,mx+1,steps):
            a = func(i,*args)
            out_dict[a] = i
        min_val = min(out_dict.keys())
        return out_dict[min_val]     
    @staticmethod
    def pdf_mnd(mu,cov,x):
        x = np.array([x])
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.pinv(cov)
        prod = -0.5 * np.matmul(np.matmul(x-mu,cov_inv),np.transpose(x-mu))
        prob = (1/(2*np.pi*cov_det)**0.5)*np.exp(prod)
        return prob
    @staticmethod
    def t_cost(nu,*args):
        e_hi = args[0]
        e_log_hi = args[1]
        nu_by_2 = nu / 2
        nu = e_hi.shape[0]*nu_by_2*math.log(nu_by_2) - gammaln(nu_by_2) + (nu_by_2 - 1)*sum(e_log_hi) - nu_by_2*sum(e_hi)
        return nu
    
    @staticmethod
    def performance_Metrics(pred,ground_truth):
        accuracy = len([x for y,x in zip(pred, ground_truth) if x==y])/len(pred)
        fal_pos = len([x for y,x in zip(pred,ground_truth) if x != y & x==1])
        fal_neg = len([x for y,x in zip(pred,ground_truth) if x != y & x==0])
        fal_pos_rate = fal_pos / 100
        fal_neg_rate = fal_neg / 100
        mis_rate = fal_pos + fal_neg
        mis_rate /= len(pred)
        return (accuracy,fal_pos_rate,fal_neg_rate,mis_rate)
if __name__ == '__main__':
    print('Local Minimum: '+repr(Utilities.line_search_min(math.sin,1,2*math.pi)))

    