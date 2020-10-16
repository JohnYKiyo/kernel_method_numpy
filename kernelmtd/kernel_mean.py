import pandas as pd

from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
from jax import jit

from .utils import transform_data
from .kernel import GaussKernel, MaternKernel

class KernelMean(object):
    """[summary]

    Args:
        object ([type]): [description]
    """    
    def __init__(self, data, kernel, weights=None): #def __init__(self, data, weights=None, kernel='Gauss',**kwargs):
        """[summary]

        Args:
            data ([type]): [description]
            weights ([type], optional): [description]. Defaults to None.
            kernel (str, optional): [description]. Defaults to 'Gauss'.

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
        """        
        if isinstance(data,pd.DataFrame):
            self._data = data
        else:
            self._data = pd.DataFrame(data)
        self._n_samples, self._n_features = self._data.shape
        self._weights = np.atleast_1d(np.full(self._n_samples,1./self._n_samples))
        if weights is not None:
            if len(weights) != self._n_samples:
                raise ValueError(f'length of weights should be {self._n_samples}')
            self._weights = np.atleast_1d(weights/np.sum(weights))
        
        self.__kernel = kernel
        #if kernel=='Gauss':
        #    if 'covariance' not in kwargs:
        #        raise ValueError('GaussKernel requires a covariance matrix as "covariance"')
        #    cov = kwargs.get('covariance')
        #    self.__kernel = GaussKernel(cov)
        #    
        #elif kernel=='Matern':
        #    a = kwargs.get('a',1.)
        #    l = kwargs.get('l')
        #    nu = kwargs.get('nu',1.5)
        #    self.__kernel = MaternKernel(a,l,nu)
        #    
        #else:
        #    raise ValueError('kernel is not defined.')
    
    def __call__(self,val,**kwargs):
        return self.kde(val,**kwargs)
    
    def kde(self,val,**kwargs):
        """[summary]

        Args:
            val ([type]): [description]

        Returns:
            [type]: [description]
        """        
        val = transform_data(val)
        kde = self.kernel.kde(val,self._data.values,**kwargs)
        return np.average(kde,weights=self._weights,axis=1)
    
    def gradkde(self,val,**kwargs):
        """[summary]

        Args:
            val ([type]): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """        
        val = transform_data(val)
        try:
            grad = self.kernel.gradkde(val,self._data.values,**kwargs)
        except:
            raise NotImplementedError
        return np.average(grad,weights=self._weights,axis=1)

    @property
    def kernel(self):
        return self.__kernel