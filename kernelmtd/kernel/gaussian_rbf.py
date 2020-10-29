from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import jit, vmap
from functools import partial
import warnings

from ..metrics.distance import euclid_distance, mahalanobis_distance
from ..utils import transform_data, pairwise, gradpairwise

#this is jit compiled distance for exec speed.
@jit
def _calculate_normalize_factor(C):
    """
    nom_factor = np.sqrt((2*pi)^k |C|)
    """
    return np.sqrt(np.linalg.det(2*np.pi*C))

def loggaussK(x1, x2, Q):
    return -0.5*mahalanobis_distance(x1,x2,Q,True)

def gauss(x1,x2,Q):
    return np.exp(-0.5*mahalanobis_distance(x1,x2,Q,True))

#loggauss_pairwise = pairwise(pairwise(loggaussK,1),1)
#gauss_pairwise = pairwise(pairwise(gauss,1),1)
#grad_gauss_pairwise = pairwise(gradpairwise(gauss,1),1)
loggauss_pairwise = pairwise(loggaussK,1)
gauss_pairwise = pairwise(gauss,1)
grad_gauss_pairwise = gradpairwise(gauss,1)

def loggauss1d(x1,x2,sigma):
    return -0.5*euclid_distance(x1,x2,True)/(sigma*sigma)

def gauss1d(x1,x2,sigma):
    return np.exp(-0.5*euclid_distance(x1,x2,True)/(sigma*sigma))

#loggauss1d_pairwise = pairwise(pairwise(loggauss1d,1),1)
#gauss1d_pairwise = pairwise(pairwise(gauss1d,1),1)
#grad_gauss1d_pairwise = pairwise(gradpairwise(gauss1d,1),1)
loggauss1d_pairwise = pairwise(loggauss1d,1)
gauss1d_pairwise = pairwise(gauss1d,1)
grad_gauss1d_pairwise = gradpairwise(gauss1d,1)

class GaussKernel(object):
    """[summary]
    The class of Gauss kernel. This class requires sigma or covariance. Specify either sigma or covariance.
    When using cov, measure the distance of the vector by Mahalanobis distance. If the variables are independent, specify the diagonal matrix.
    Even if the size of cov is 1000 * 1000 or less, inverse matrix calculation is not possible.
    When using sigma, measure the distance of the vector by Euclidean distance.
    
    Args:
        covariance (ndarray, optional): Defaults to None.
            A scale used to measure the distance between vectors. Multidimensional version of kernel bandwidth. 
            
        sigma (scalar, optional): Defaults to None.
            A scale used to measure the distance between vectors. Simple gauss kernel bandwidth.
            
    Raises:
        ValueError: Require that either the covariance matrix (covariance) \"or\" the bandwidth (sigma) be defined.
        ValueError: Covariance should be numpy.ndarray.
        ValueError: The size of covariance matrix is huge.
        ValueError: Sigma should be scalar.
    
    Examples:
        >>> import numpy as onp
        >>> kernel_gauss = GaussKernel(sigma=1.)
        >>> x = np.atleast_2d(np.linspace(-5.,5.,10)).T
        >>> onp.asarray(kernel_gauss.kde(x,np.array([[0.],[1.]])))
        array([[3.72665317e-06, 1.52299797e-08],
               [5.19975743e-04, 6.45524691e-06],
               [2.11096565e-02, 7.96086691e-04],
               [2.49352209e-01, 2.85655008e-02],
               [8.56996891e-01, 2.98234096e-01],
               [8.56996891e-01, 9.05955191e-01],
               [2.49352209e-01, 8.00737403e-01],
               [2.11096565e-02, 2.05924246e-01],
               [5.19975743e-04, 1.54084456e-02],
               [3.72665317e-06, 3.35462628e-04]])
    """  
    def __init__(self, covariance=None,sigma=None):
        if ((covariance is None) and (sigma is None)) or ((covariance is not None) and (sigma is not None)):
            raise ValueError('Require that either the covariance matrix (covariance) \"or\" the bandwidth (sigma) be defined.')
        
        if covariance is not None:
            if not isinstance(covariance,np.ndarray):
                raise ValueError('covariance should be numpy.ndarray.')
            if covariance.size > 1000000:
                raise ValueError('\
                    !! The size of covariance matrix is huge. !! \n \
                    Inverse matrix may not be calculated')
            self.__n_features = covariance.shape[1]
            self.__cov = np.atleast_2d(covariance).astype(float)
            self.__inv_cov = np.linalg.inv(covariance)
            self.__sigma = sigma
            self.__inv_sigma = None
            self.__kde = jit(partial(lambda x1,x2,Q: gauss_pairwise(x1,x2,Q),Q=self.__inv_cov))
            self.__logkde = jit(partial(lambda x1,x2,Q: loggauss_pairwise(x1,x2,Q),Q=self.__inv_cov))
            self.__grad_kde = jit(partial(lambda x1,x2,Q: grad_gauss_pairwise(x1,x2,Q),Q=self.__inv_cov)) 
            self.__norm_factor = _calculate_normalize_factor(self.__cov)

        if sigma is not None:
            if not (np.isscalar(sigma) or sigma.size==1):
                raise ValueError('sigma should be scalar !')
            self.__n_features = None
            self.__cov = covariance
            self.__inv_cov = None
            self.__sigma = float(sigma)
            self.__inv_sigma = 1./sigma
            self.__kde = jit(partial(lambda x1,x2,sig: gauss1d_pairwise(x1,x2,sig),sig=self.__sigma))
            self.__logkde = jit(partial(lambda x1,x2,sig: loggauss1d_pairwise(x1,x2,sig),sig=self.__sigma))
            self.__grad_kde = jit(partial(lambda x1,x2,sig: grad_gauss1d_pairwise(x1,x2,sig),sig=self.__sigma))
            self.__norm_factor = np.sqrt(2*np.pi)*self.__sigma
    
    def __call__(self,x1,x2,**kwargs):
        ##　Other kernels may not have a normalize option, so use kwargs.
        normalize = kwargs.get('normalize',False)
        return self.kde(x1,x2,normalize=normalize)
    
    def __repr__(self):
        val =  f'Kernel: GaussKernel\n'
        val += f'Normalization factor: {self.norm_factor}\n'
        val += f'Covariance matrix: \n{self.cov}\n'
        val += f'Sigma (bandwidth):{self.sigma}\n'
        val += f'Dimensions: {self.n_features}\n'
        return val
    
    def logkde(self,x1,x2, **kwargs):
        normalize = kwargs.get('normalize',False)
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        if self.__cov is not None: 
            if (x1.shape[1] != self.__n_features) or (x2.shape[1] != self.__n_features):
                raise ValueError(f"The features dimention (x1,x2):({x1.shape[1]},{x2.shape[1]})" \
                                 f"should be same as (n_features,n_features).")

        val= self.__logkde(x1,x2)
        if normalize:
            val += -1.*np.log(self.__norm_factor)
        return val
    
    def kde(self,x1,x2,**kwargs):
        normalize = kwargs.get('normalize',False)
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        if self.__cov is not None: 
            if (x1.shape[1] != self.__n_features) or (x2.shape[1] != self.__n_features):
                raise ValueError(f"The features dimention (x1,x2):({x1.shape[1]},{x2.shape[1]})" \
                                 f"should be same as (n_features,n_features).")
            
        val= self.__kde(x1,x2)
        if normalize:
            val = val/self.__norm_factor
        return val
    
    def gradkde(self,x1,x2,**kwargs):
        normalize = kwargs.get('normalize',False) 
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        if self.__cov is not None: 
            if (x1.shape[1] != self.__n_features) or (x2.shape[1] != self.__n_features):
                raise ValueError(f"The features dimention (x1,x2):({x1.shape[1]},{x2.shape[1]})" \
                                 f"should be same as (n_features,n_features).")

        val = self.__grad_kde(x1,x2)
        if normalize:
            val = val/self.__norm_factor
        return val

    @property
    def norm_factor(self):
        return self.__norm_factor
    
    @property
    def cov(self):
        return self.__cov
    
    @cov.setter
    def cov(self, covariance):
        self.__init__(covariance=covariance)
    
    @property
    def inv_cov(self):
        return self.__inv_cov
    
    @property
    def sigma(self):
        return self.__sigma
    
    @sigma.setter
    def sigma(self,sigma):
        self.__init__(sigma=sigma)
    
    @property
    def inv_sigma(self):
        return self.__inv_sigma
    
    @property
    def n_features(self):
        return self.__n_features