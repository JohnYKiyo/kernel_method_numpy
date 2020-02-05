from .utils.functions import gauss_kernel

import numpy as np
import pandas as pd

class KernelMean():
    """ 
    input: data is pandas dataframe
    """
    def __init__(self, data, cov, weights=None):
        if isinstance(data,pd.DataFrame):
            self.data = data
        else:
            self.data = pd.DataFrame(data)
        self._n_samples, self._n_features = self.data.shape
        self._cov = cov
        
        self._weights = np.atleast_1d(np.full(self._n_samples,1./self._n_samples))
        if weights is not None:
            if len(weights) != self._n_samples:
                raise ValueError(f'length of weights should be {self._n_samples}')
            self._weights = np.atleast_1d(weights/np.sum(weights))

        self.kernel = gauss_kernel.gauss_kernel(covariance=self._cov, n_features=self._n_features)

    @property
    def weights(self):
        return self._weights

    def kernel_mean(self, val,normalize=False):
        """
        input shuld be numpy 2d-array. np.array([[x,x,x,x], [...] ,... ,[...]])
        """
        val = np.atleast_2d(val)
        return np.average(self.kernel.pdf(x=val, y=self.data, normalize=normalize),weights=self._weights, axis=1)
        #return np.dot(self.kernel.pdf(x=val, y=self.x, normalize=True), np.atleast_2d(self._weights).T) #-> (NxF) x (Fx1)=(Nx1)
    
    def grad_kernel_mean(self, val, normalize=False):
        val = np.atleast_2d(val)
        return np.average(self.kernel.grad(x=val,y=self.data,normalize=normalize), weights=self._weights, axis=1)

    def __call__(self,val):
        return self.kernel_mean(val)
