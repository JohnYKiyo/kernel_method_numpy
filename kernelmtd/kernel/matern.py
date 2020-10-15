from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import jit, vmap

import warnings

from kernelmtd.metrics.distance import euclid_distance
from kernelmtd.utils import transform_data, pairwise

@jit
def K_0p5(x1,x2,l,nu):
    dists = pairwise(euclid_distance, square=False)
    return np.exp(-dists(x1,x2)/l)

@jit
def K_1p5(x1,x2,l,nu):
    dists = pairwise(euclid_distance, square=False)
    K = dists(x1,x2)/l * np.sqrt(3)
    return (1. + K) * np.exp(-K)

@jit
def K_2p5(x1,x2,l,nu):
    dists = pairwise(euclid_distance, square=False)
    K = dists(x1,x2)/l * np.sqrt(5)
    return (1. + K + K ** 2 / 3.0) * np.exp(-K)

@jit
def K_inf(x1,x2,l,nu):
    dists = pairwise(euclid_distance, square=True)
    return np.exp(-dists(x1,x2) / 2.0 /l**2)

def K_other(x1,x2,l,nu):
    dists = pairwise(euclid_distance, square=False)
    dists_matrix = dists(x1,x2)/l
    dists_matrix = np.where(dists_matrix==0, np.finfo(float).eps, dists_matrix)
    tmp = (np.sqrt(2 * nu) * dists_matrix)
    val = (2 ** (1. - nu)) / np.exp(scp.special.gammaln(nu))
    return val * tmp**nu * kv(nu,tmp)

def matern(x,y, l=1., nu=1.5):
    if nu == 0.5:
        return K_0p5(x,y,l,nu)
    elif nu == 1.5:
        return K_1p5(x,y,l,nu)
    
    elif nu == 2.5:
        return K_2p5(x,y,l,nu)
    
    elif nu == np.inf:
        return K_inf(x,y,l,nu)
    else:
        warnings.warn('Slow processing speed', FutureWarning)
        return K_other(x,y,l,nu)

class MaternKernel(object):
    def __init__(self, a=1.0, l=1.0, nu=1.5, *args,**kwargs):
        self.__a = a
        self.__l = l
        self.__nu = nu

    def __call__(self,x1,x2,**kwargs):
        return self.kde(x1,x2)
    
    def kde(self,x1,x2,**kwargs):
        return self.__a*matern(x1,x2, self.__l, self.__nu)
    
    def logkde(self,x1,x2,**kwargs):
        return np.log(self.__a*matern(x1,x2, self.__l, self.__nu))
    
    @property
    def a(self):
        return self.__a

    @property
    def l(self):
        return self.__l

    @property
    def nu(self):
        return self.__nu