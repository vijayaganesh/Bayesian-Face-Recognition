# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 17:00:13 2018

@author: VijayaGanesh Mohan
"""
import numpy as np
from scipy.special import digamma,gammaln
import os, sys,math
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)
from utilities.Utilities import Utilities
from common.model import Model

class T_Dist(Model):
    def __init__(self,records):
        Model.__init__(self,records)
        self.mu = np.random.permutation(records)[0]
        self.cov = np.cov(records.T)
        self.cov = np.diag(np.diag(self.cov))
        self.nu = 1000
    
    def compute_cost_function(self):
        e_hi = np.zeros(self.nrows)
        e_log_hi = np.zeros(self.nrows)
        cov_inv = np.linalg.pinv(self.cov)
        for i in range(0,self.nrows-1):
            comp_term = np.matmul(np.matmul(self.x[i]-self.mu,cov_inv),np.transpose(self.x[i]-self.mu))
            e_hi[i] = (self.nu+self.dimensions)/(self.nu+comp_term)
            e_log_hi[i] = digamma((self.nu+self.dimensions)/2) - np.log((self.nu + comp_term)/2)
        return (e_hi,e_log_hi)
    
    def update_vars(self,cost_function):
        e_hi,e_log_hi = cost_function
        self.mu = np.zeros(self.dimensions)
        for i in range(self.nrows):
            self.mu += e_hi[i]*self.x[i]
        self.mu /= sum(e_hi)
        x_minus_mean = self.x - self.mu
        self.cov = np.zeros((self.dimensions,self.dimensions))
        for i in range(0,self.nrows-1):
            self.cov +=  e_hi[i]*np.matmul(x_minus_mean[i], x_minus_mean[i].T)
        self.cov /= sum(e_hi)
        self.cov = np.diag(np.diag(self.cov))
        self.nu = Utilities.line_search_min(Utilities.t_cost,0.1,1000,10000,e_hi,e_log_hi)
        return self.compute_cost_function()
            
    def exp_max(self,tolerance):
        prev_cost = -np.inf
        e_hi,e_log_hi = self.compute_cost_function()
        cov_inv = np.linalg.pinv(self.cov)
        comp_term = np.zeros(self.nrows)
        for i in range(0,self.nrows-1):
            comp_term[i] = np.matmul(np.matmul(self.x[i]-self.mu,cov_inv),np.transpose(self.x[i]-self.mu))
        curr_cost = self.nrows*gammaln((self.nu+self.dimensions)/2) - (self.dimensions/2) * (np.log(self.nu * math.pi)) - np.log(np.linalg.det(self.cov))/2 - gammaln(self.nu/2)
        curr_cost  -= (self.nu+self.dimensions) * sum(np.log(1 + comp_term/self.nu)) / 2    
        initial_cost = curr_cost
        iteration = 0
        print(curr_cost)
        while((curr_cost - prev_cost) > tolerance ):
            iteration += 1
            print("Expectation Maximization Iteration: "+repr(iteration))
            prev_cost = curr_cost 
            e_hi,e_log_hi = self.update_vars((e_hi,e_log_hi))
            cov_inv = np.linalg.pinv(self.cov)
            comp_term = np.zeros(self.nrows)
            for i in range(0,self.nrows-1):
                comp_term[i] = np.matmul(np.matmul(self.x[i]-self.mu,cov_inv),np.transpose(self.x[i]-self.mu))
            curr_cost = self.nrows*gammaln((self.nu+self.dimensions)/2) - (self.dimensions/2) * (np.log(self.nu * math.pi)) - np.log(np.linalg.det(self.cov))/2 - gammaln(self.nu/2)
            curr_cost  -= (self.nu+self.dimensions) * sum(np.log(1 + comp_term/self.nu)) / 2
            e_hi,e_log_hi = self.compute_cost_function()
            print("p_cost: "+ repr(prev_cost))
            print("c_cost: "+ repr(curr_cost))
        print("EM Complete in "+repr(iteration)+" iterations.")
        print("Initial Cost: "+repr(np.sum(initial_cost)))
        print("Final Cost: "+repr(np.sum(curr_cost)))