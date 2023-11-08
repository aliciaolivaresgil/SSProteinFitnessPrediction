"""
This class implements the hybrid model defined in https://www.biorxiv.org/content/10.1101/2022.06.07.495081v1.abstract. 
Author: Alicia Olivares-Gil
"""

from sklearn.base import BaseEstimator

import pickle
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from scipy.optimize import differential_evolution

from sklearn.linear_model import Ridge

class Merge(BaseEstimator): 
    
    def __init__(self, wild_type, alpha=0.9): 
        
        self.wild_type = wild_type
        self.alpha = alpha
        self.beta_1 = None
        self.beta_2 = None
        self.ridge = None
        
        self._spearmanr_dca = None
    
    
    def fit(self, X_train, y_train): 
        
        #Spearman's rank correlation coefficient of the (full) data and the DCA predictions (difference of statistical energies)
        self._spearmanr_dca = self._calc_spearmanr_dca(X_train, y_train)
        
        #Predictions of the statistical method to adjust betas
        y_dca = self._delta_E(X_train)
        
        #Predictions of the linear method to adjust betas 
        self.ridge = Ridge(alpha=self.alpha).fit(X_train, y_train)
        y_ridge = self.ridge.predict(X_train)
        
        #adjust betas 
        self.beta_1, self.beta_2 = self._adjust_betas(y_train, y_dca, y_ridge)

    
    def predict(self, X_test): 

        y_dca = self._delta_E(X_test)
        y_ridge = self.ridge.predict(X_test)
        
        if self._spearmanr_dca >= 0: 
            return self.beta_1*y_dca + self.beta_2*y_ridge
        else: 
            return self.beta_1*y_dca - self.beta_2*y_ridge
    
    
    def score(self, X_test, y_test): 
        return spearmanr(y_test, self.predict(X_test))[0]
    
    
    def _delta_X(self, X):
        return np.subtract(X, self.wild_type)
    
    
    def _delta_E(self, X): 
        return np.sum(self._delta_X(X), axis=1)
    
    
    def _calc_spearmanr_dca(self, X, y): 
        y_dca = self._delta_E(X)
        return spearmanr(y, y_dca)[0]
    
    
    def _adjust_betas(self, y, y_dca, y_ridge): 
        loss = lambda b: -np.abs(spearmanr(y, b[0]*y_dca + b[1]*y_ridge)[0])
        minimizer= differential_evolution(loss, bounds=[(0,1),(0,1)], tol=1e-4)
        return minimizer.x
    
    
    
    
    