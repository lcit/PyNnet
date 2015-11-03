'''
Neural network: evaluation class
Leonardo Citraro
'''
from __future__ import division # impose float division
import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
from sklearn import metrics

class Evaluator(object):
    
    def __init__(self, score_types=['accuracy'], sample_weight=None):
        self.score_types    = score_types # one or multiple score to compute here
        self.sample_weight  = sample_weight # weights for each sample
    
    def _compute_single( self, score_type, y, y_hat):
        #--------------------------------------------------------------------------
        if score_type == "accuracy":
            return metrics.accuracy_score(  y, 
                                            y_hat>0.5, 
                                            sample_weight=self.sample_weight, 
                                            normalize=True  )
        #--------------------------------------------------------------------------                                    
        elif score_type == "f1_score":
            return metrics.f1_score(    y, 
                                        y_hat>0.5, 
                                        sample_weight=self.sample_weight     )
        #--------------------------------------------------------------------------
        elif score_type == "auc":
            return metrics.roc_auc_score(   y, 
                                            y_hat, 
                                            sample_weight=self.sample_weight     )
        #--------------------------------------------------------------------------
        elif score_type == "log-loss":
            epsilon = 1e-15
            pred = sp.maximum(epsilon, y_hat)
            pred = sp.minimum(1-epsilon, pred)
            if self.sample_weight is None:
                J = np.sum(     - y*sp.log(pred) \
                                - sp.subtract(1,y)*sp.log(sp.subtract(1,pred))) \
                                /y.shape[0]
            else:
                J = np.sum(     - y*sp.log(pred)*self.sample_weight \
                                - sp.subtract(1,y)*sp.log(sp.subtract(1,pred)) \
                                *self.sample_weight)/y.shape[0]
            return J
        #--------------------------------------------------------------------------
        elif score_type == "quadratic-loss":
            if self.sample_weight is None:
                J = 0.5*np.sum((y-y_hat)**2)/y.shape[0]
            else:
                J = 0.5*np.sum((self.sample_weight*(y-y_hat))**2) \
                            /y.shape[0]
            return J
        #--------------------------------------------------------------------------
        else:
            raise ValueError('Evaluator: undefined score_type.')
    
    def compute_score( self, y, y_hat):
        # check if score_types is a list
        if isinstance(self.score_types, list):     
            return [self._compute_single( type, y, y_hat) for type in self.score_types]
        else:
            return self._compute_single( self.score_types, y, y_hat)

            
            
            
            
            
            
            
            
            
            
            