from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import jit, vmap

import warnings

from kernelmtd.metrics.distance import euclid_distance, mahalanobis_distance
from kernelmtd.utils import transform_data, pairwise, gradpairwise

#this is jit compiled distance for exec speed.
@jit
def _calculate_normalize_factor(C):
    """
    nom_factor = np.sqrt((2*pi)^k |C|)
    """
    return np.sqrt(np.linalg.det(2*np.pi*C))

class GaussKernel(object):
    def __init__(self, covariance):
        self.__cov = covariance
        self.__inv_cov = np.linalg.inv(covariance)
        self.__n_features = covariance.shape[1]
        self.__dists = pairwise(mahalanobis_distance,Q=self.__inv_cov,square=True)
        self.__grad_dists = gradpairwise(mahalanobis_distance,Q=self.__inv_cov,square=True)
        try:
            self.__norm_factor = _calculate_normalize_factor(self.__cov)
        except:
            warnings.warn('The normalization factor could not be calculated. set to 1.')
            self.__norm_factor = 1.
    
    def __call__(self,x1,x2,**kwargs):
        ##　Other kernels may not have a normalize option, so use kwargs.
        normalize = kwargs.get('normalize',False)
        return self.kde(x1,x2,normalize=normalize)
    
    def logkde(self,x1,x2, **kwargs):
        normalize = kwargs.get('normalize',False)
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        if (x1.shape[1] != self.__n_features) or (x2.shape[1] != self.__n_features):
            raise ValueError(f"The features dimention (x1,x2):({x1.shape[1]},{x2.shape[1]})" \
                             f"should be same as ({self.__n_features},{self.__n_features}).")
        
        val= -0.5*self.__dists(x1,x2)
        if normalize:
            val += -1.*np.log(self.__norm_factor)
        return val
    
    def kde(self,x1,x2,**kwargs):
        normalize = kwargs.get('normalize',False)
        return np.exp(self.logkde(x1,x2,normalize=normalize))
    
    def gradkde(self,x1,x2,**kwargs):
        normalize = kwargs.get('normalize',False) 
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        return np.einsum('ij,ijk->ijk',
                         np.exp(self.logkde(x1,x2,normalize=normalize)),
                         (-0.5*self.__grad_dists(x1,x2)))
    
    @property
    def norm_factor(self):
        return self.__norm_factor
    
    @property
    def cov(self):
        return self.__cov
    
    @cov.setter
    def cov(self, covariance):
        self.__init__(covariance)
    
    @property
    def inv_cov(self):
        return self.__inv_cov
    
    @property
    def n_features(self):
        return self.__n_features