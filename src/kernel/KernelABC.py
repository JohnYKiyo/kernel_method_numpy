from .utils import ABCDataSet
from .utils.functions import band_width, gauss_kernel, kernel_gram_matrix
from .KernelMean import KernelMean

import numpy as np
import pandas as pd

class KernelABC():
    def __init__(self,Dataset,cov_y=None,cov_para=None):
        if not isinstance(Dataset,ABCDataSet):
            TypeError(f'Type is not ABCDataSet type.')

        self.Dataset = Dataset
        self.n_para_set = self.Dataset.parameters.shape[0]
        self._epsilon = 0.01/np.sqrt(self.n_para_set)

        self.cov_y = cov_y
        if isinstance(cov_y,str):
            bw_y = band_width(self.Dataset.prior_data.values, method=cov_y)
            self.cov_y = bw_y.cov.copy()

        self.cov_para = cov_para 
        if isinstance(cov_para,str):
            bw_para = band_width(self.Dataset.parameters.values, method=cov_para)
            self.cov_para = bw_para.cov.copy()

        self.kernel_y = kernel_gram_matrix(self.Dataset.prior_data.values,self.cov_y)

        self._kernel_ridge_regression()

    def _kernel_ridge_regression(self):
        G_NeI = self.kernel_y.gram_matrix + self.n_para_set*self._epsilon*np.eye(self.n_para_set)
        k_y = self.kernel_y.pdf(self.Dataset.prior_data.values,
                                self.Dataset.observed_samples.values,False)
        w = np.dot(np.linalg.inv(G_NeI), k_y)
        w = w/w.sum()
        self._w = w
        self._G_NeI = G_NeI
        self._k_key = k_y
        self._post_para_kernel = KernelMean(data=self.Dataset.parameters,
                                            cov=self.cov_para,
                                            weights=self._w.squeeze()) 

    def posterior_mean(self):
        return pd.DataFrame(np.dot(self._w.T,self.Dataset.parameters.values),
                            columns=self.Dataset.parameter_keys,index=['mean'])

    def posterior_kernel_mean(self,data):
        return self._post_para_kernel.kernel_mean(data)
    
    @property
    def posterior_kernel(self):
        return self._post_para_kernel

    @property
    def weights(self):
        return self._w