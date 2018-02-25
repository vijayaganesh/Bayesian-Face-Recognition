# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 17:57:13 2018

@author: VijayaGanesh Mohan
"""
import numpy as np
import os, sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)
from utilities.Utilities import Utilities
from common.model import Model
class MoG(Model):
    def __init__(self,records,mixtures):
        Model.__init__(self,records,mixtures)
        self.theta = np.ones(mixtures)*(1/mixtures)
    
    def compute_cost_function(self):
        k = len(self.theta)
        r = np.ones((self.nrows,k))
        for i in range(0,self.nrows):
            input_x = self.x[i]
            sum_lh = 0
            for j in range(0,k):
                r[i][j] = self.theta[j] * Utilities.pdf_mnd(self.mu[j],np.diag(np.diag(self.cov[j])),input_x)
                sum_lh += r[i][j]
            r[i] /= sum_lh
        return r
    
    def update_vars(self,cost_function):
        self.theta = sum(cost_function)/np.sum(cost_function)
        for k in range(0,len(self.theta)):
            mu_num = np.zeros(self.dimensions)
            for i in range(0,self.nrows):
                mu_num += cost_function[i][k]*self.x[i]
            self.mu[k] = mu_num / sum(cost_function)[k]
            cov_num = np.zeros((self.dimensions,self.dimensions))
            for i in range(0,self.nrows):
                cov_num += cost_function[i][k]*((self.x[i]-self.mu[k]).T)*(self.x[i]-self.mu[k])
            self.cov[k] = cov_num / sum(cost_function)[k]
        return self.compute_cost_function()
            
    def exp_max(self,tolerance):
        prev_cost = float('inf')
        curr_cost = self.compute_cost_function()
        initial_cost = curr_cost
        iteration = 0
        while(abs(np.sum(prev_cost) - np.sum(curr_cost)) > tolerance):
            iteration += 1
            print("Expectation Maximization Iteration: "+repr(iteration))
            prev_cost = curr_cost
            curr_cost = self.update_vars(curr_cost)
        print("EM Complete in "+repr(iteration)+" iterations.")
        print("Initial Cost: "+repr(np.sum(initial_cost)))
        print("Final Cost: "+repr(np.sum(curr_cost)))
    
    
    
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        