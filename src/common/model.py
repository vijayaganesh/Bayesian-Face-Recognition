# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:20:34 2018

@author: VijayaGanesh Mohan
@email: vmohan2@ncsu.edu

"""
from abc import ABC, abstractmethod
import numpy as np
np.random.seed(5)
class Model(ABC):
    
    def __init__(self,records,mixtures):
        self.dimensions = records.shape[1]
        self.nrows = records.shape[0]
        self.mu = np.random.permutation(records)[0:mixtures]
        self.cov = np.ones((mixtures,self.dimensions,self.dimensions))*np.cov(records.T)
        self.x = records
    @abstractmethod
    def compute_cost_function(self):
        pass
    @abstractmethod
    def update_vars(self):
        pass
    @abstractmethod
    def exp_max(self):
        pass
    