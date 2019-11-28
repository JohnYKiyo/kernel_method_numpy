from .utils import ABCDataSet
from .utils.functions import get_band_width, gauss_kernel, gram_matrix
from .KernelMean import KernelMean

import numpy as np
import pandas as pd

class KernelABC():
    def __init__(self,Dataset,sigma_y=None,sigma_para=None):
        if not isinstance(Dataset,ABCDataSet):
            TypeError(f'Type is not ABCDataSet type.')
        
        self.Dataset = Dataset
        self.sigma_y = sigma_y
        if isinstance(sigma_y,str):
            self.sigma_y = get_band_width(self.Dataset.prior_data.values,method=sigma_y)

        self.sigma_para = sigma_para
        if isinstance(sigma_para,str):
            self.sigma_para = get_band_width(self.Dataset.parameters.values,method=sigma_para)
        
        self.kernel = gauss_kernel(self.sigma_y)
        self.n_theta_set = self.Dataset.parameters.shape[0]
        self.epsilon = 0.01/np.sqrt(self.n_theta_set)
        self.gram = gram_matrix(self.Dataset.prior_data.values,self.sigma_y)
        
        self._kernel_ridge_regression()
        
    def _kernel_ridge_regression(self):
        G_NeI = self.gram.gram_matrix + self.n_theta_set*self.epsilon*np.eye(self.n_theta_set)
        k_y = self.kernel.compute(self.Dataset.prior_data.values,
                                  self.Dataset.observed_samples.values,False)
        w = np.dot(np.linalg.inv(G_NeI), k_y)
        w = w/w.sum()
        self.w = w
        self.G_NeI = G_NeI
        self.k_key = k_y
        
    def posterior_mean(self):
        return pd.DataFrame(np.dot(self.w.T,self.Dataset.parameters.values),
                            columns=self.Dataset.parameter_keys,index=['mean'])
    
    def posterior_kernel(self):
        return KernelMean(sigma=self.sigma_para,weights=self.w.squeeze(),x=self.Dataset.parameters)
