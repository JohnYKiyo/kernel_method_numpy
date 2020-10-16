import pandas as pd

from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
from jax import jit

from .utils import transform_data
from .kernel import GaussKernel, MaternKernel
from .kernel_mean import KernelMean

class KernelABC(object):
    def __init__(self,data,para,kernel,epsilon=0.01):
        if data.shape[0] != para.shape[0]:
            error = f'Data and parameters must have a one-to-one correspondence.'\
                  + f'but the number of data:{data.shape[0]}, para:{para.shape[0]}'
            raise ValueError(error)
        self.__data = data
        self.__para = para
        self.__epsilon = epsilon
        self.__ndim_data = data.shape[1]
        self.__ndim_para = para.shape[1]
        self.__ndata = data.shape[0]
        
        self.__kernel = kernel

        self.__calculate_gram_matrix()
        
    def __calculate_gram_matrix(self):
        G = self.__kernel(self.__data,self.__data) + self.__ndata * self.__epsilon * np.eye(self.__ndata)
        self.__gram_matrix = G
        self.__gram_inv = np.linalg.inv(G)
    
    def conditioning(self,obs):
        if obs.shape[1] != self.__data.shape[1]:
            error = f'Observed data should be same dimension of "data",'\
                  + f'but got obs dim:{obs.shape[1]}, data dim:{self.__data.shape[1]}'
            raise ValueError(error)
        self.__obs = obs
        self.__k_obs = np.prod(self.__kernel(self.__data,self.__obs,normalize=False),axis=1,keepdims=True) #(ndata,1) vector
        w = np.dot(self.__gram_inv,self.__k_obs)
        self.__weights = w.ravel()
        print(f'sum weights:{w.sum()}')
    
    def expected_param(self):
        """
        E_{\theta|Y_{obs}}[\theta] = <m_{\theta}|Y_{obs} | \cdot > = \sum_i w_i \theta_i
        """
        return np.average(self.__para, axis=0, weights=self.__weights)
    
    def posterior_kernel_mean(self,kernel):
        return KernelMean(data=self.__para, kernel=kernel, weights=self.__weights)
    
    @property
    def gram(self):
        return self.__gram_matrix
    @property
    def kernel(self):
        return self.__kernel
    @property
    def ndim_data(self):
        return self.__ndim_data
    @property
    def ndim_para(self):
        return self.__ndim_para
    @property
    def data(self):
        return self.__data
    @property
    def para(self):
        return self.__para
    @property
    def weights(self):
        return self.__weights
    @property
    def k_obs(self):
        return self.__k_obs