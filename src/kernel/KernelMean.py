from .utils.functions import get_band_width, gauss_kernel

import numpy as np
import pandas as pd

class KernelMean():
    """ 
    input: x is pandas dataframe
    """
    def __init__(self, x, sigma='median', weights=None):
        if isinstance(x,pd.DataFrame):
            self.x = x 
        else:
            self.x = pd.DataFrame(x)
            
        self.sigma = sigma
        if isinstance(sigma,str):
            self.sigma = get_band_width(self.x.values,sigma)
            
        self.p = np.array([np.full(len(self.x),1./len(self.x))])
        if weights is not None:
            self.p = np.array([weights/np.sum(weights)])
                    
        self.mu_p = lambda val: self._compute_kernel_from_samples(val)
          
    def _compute_kernel_from_samples(self, val):
        """
        input shuld be numpy 2d-array. np.array([[x,x,x,x], [...] ,... ,[...]])
        """
        kernel = gauss_kernel(self.sigma)
        weighted_val = np.dot(kernel(val,self.x),self.p.T)
        return weighted_val
