# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:33:22 2018

@author: VijayaGanesh Mohan
@email: vmohan2@ncsu.edu

"""

import numpy as np
import os, sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)
from common.model import Model
class Factor_Analyzer(Model):
    def __init__(self,records,factors):
        Model.__init__(self,records)
        self.factors = factors
        self.mu = np.mean(records,0)
        self.cov = np.diag(np.cov(records.T))
        self.phi = np.random.rand(self.dimensions,factors)
    
    def compute_cost_function(self):
        p_t_s_i = self.phi.T * (1 / self.cov)
        e_hi = np.matmul(np.linalg.pinv(np.matmul(p_t_s_i, self.phi)),p_t_s_i).dot(np.array(self.x-self.mu).T)
        print(e_hi.shape)
        e_hi_hi_t = []
        for i in range(self.nrows):
            e_hi_hi_t.append(np.linalg.pinv(p_t_s_i.dot(self.phi)) + np.matmul(e_hi[:,],e_hi[:,].T))
        e_hi_hi_t = np.array(e_hi_hi_t)
        return (e_hi,e_hi_hi_t)
    
    def update_vars(self,cost_function):
        e_hi,e_hi_hi_t = cost_function 
        phi_1 = np.zeros((self.dimensions,self.factors))
        phi_2 = np.zeros((self.factors,self.factors))
        for i in range(self.nrows):
            phi_1 = phi_1 + np.matmul(np.array([self.x[i]-self.mu]).T, np.array([e_hi[:,i]]))
            phi_2 = phi_2 + e_hi_hi_t[i]
        print(phi_1.shape)
        print(phi_2.shape)
        self.phi = np.matmul(phi_1,np.linalg.pinv(phi_2))
        self.cov = np.zeros((self.dimensions))
        print((np.matmul(self.phi,e_hi[:,2]) * (self.x[2]-self.mu).T).shape)
        for i in range(self.nrows):
            xm = self.x[i] - self.mu
            self.cov += xm.dot(xm)
            self.cov -= np.matmul(self.phi,e_hi[:,i])*xm.T
        self.cov /= self.nrows
    def exp_max(self,iterations):
        for i in range(1,iterations+1):
            print("EM Iteration: "+repr(i))
            cost = self.compute_cost_function()
            self.update_vars(cost)
        print("EM Complete for "+repr(iterations)+" iterations.")
        