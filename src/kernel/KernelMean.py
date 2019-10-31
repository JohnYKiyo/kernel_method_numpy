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
            self.sigma = get_band_width(x.values,sigma)
        
        self.p = np.full(len(x),1./len(x))
        if weights is not None:
            self.p = weights/np.sum(weights)
                
        self.mu_p = lambda val: self._compute_kernel_from_samples(val)
      
    def _compute_kernel_from_samples(self, val):
        kernel = gauss_kernel(self.sigma)
        return_val = 0
        for sample, p in zip(self.x.values,self.p):
            return_val += p*kernel(val,sample)
        return return_val
