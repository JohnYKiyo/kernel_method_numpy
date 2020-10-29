import pandas as pd

from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
from jax import jit

from .utils import transform_data
from .kernel import GaussKernel, MaternKernel
from .kernel_mean import KernelMean

class KernelABC(object):
    """[summary]
    The class of Kernel Mean.
    
    Args:
        data (array-like): 
            The shape of array should be (n_samples, n_data_dim).
            The output value that came out of the model.
            Conditioning is done on this data. :math:`P(para|data=obs)`.
            Model parameters (para) and model outputs (data) indexes need to be aligned.
            
        para (array-like): 
            The shape of array should be (n_samples, n_para_dim).
            Parameters used when outputting data from model.
            
        kernel (class kernel): 
            Specify an instance of kernel class as shown in Example.
            
        epsilon (float, optional): Regularization parameter. Defaults to 0.01.

    Raises:
        ValueError: Occurs when the number of samples of para and the number of samples of data do not match.
        
    Note:
        [1] Fukumizu, K., Song, L., & Gretton, A. (2013). Kernel Bayes’ rule: Bayesian inference with positive definite kernels. Journal of Machine Learning Research, 14, 3753–3783.
        [2] Nakagome, S., Fukumizu, K., & Mano, S. (2013). Kernel approximate Bayesian computation in population genetic inferences. Statistical Applications in Genetics and Molecular Biology, 12(6), 667–678. https://doi.org/10.1515/sagmb-2012-0050
    
    """      
    def __init__(self,data,para,kernel,epsilon=0.01):
        if data.shape[0] != para.shape[0]:
            error = f'Data and parameters must have a one-to-one correspondence.'\
                  + f'but the number of data:{data.shape[0]}, para:{para.shape[0]}'
            raise ValueError(error)
        self.__data = transform_data(data)
        self.__para = transform_data(para)
        self.__epsilon = epsilon
        self.__ndim_data = data.shape[1]
        self.__ndim_para = para.shape[1]
        self.__ndata = data.shape[0]
        
        self.__kernel = kernel

        self.__calculate_gram_matrix()
        
    def __calculate_gram_matrix(self):
        G = self.__kernel(self.__data,self.__data) 
        regularization = self.__ndata * self.__epsilon * np.eye(self.__ndata)
        self.__gram_matrix = G
        self.__gram_with_regularization = G+regularization
        self.__gram_inv = np.linalg.inv(self.__gram_with_regularization)
    
    def conditioning(self,obs):
        """[summary]
        Compute conditional kernel with obs.
        :math:`m_P(para|data = obs)`
        
        Args:
            obs (array-like):
                The shape of array should be (n_samples, n_data_dim).
        """        
        self.__obs = transform_data(obs)
        if obs.shape[1] != self.__data.shape[1]:
            error = f'Observed data should be same dimension of "data",'\
                  + f'but got obs dim:{obs.shape[1]}, data dim:{self.__data.shape[1]}'
            raise ValueError(error)
        self.__k_obs = np.prod(self.__kernel(self.__data,self.__obs,normalize=False),axis=1,keepdims=True) #(ndata,1) vector
        self.__weights = np.dot(self.__gram_inv,self.__k_obs)
        print(f'sum weights:{self.__weights.sum()}')
    
    def posterior_kernel_mean(self,kernel):
        """[summary]
        Returns the posterior kernel mean of para which is class instance of KernelMean
        Args:
            kernel (class kernel): 
                Specify an instance of kernel class.
                para
        Returns:
            KernelMean: the class of kernel mean
        """        
        return KernelMean(data=self.__para, kernel=kernel, weights=self.__weights)
    
    @property
    def eap(self):
        """
        Expected A Posteriori
        :math:`E_{\\theta|Y_{obs}}[\\theta] = < m_{\\theta}|Y_{obs} | \\cdot > = \\sum_i w_i \\theta_i`.
        """
        EAP= np.average(self.__para, axis=0, weights=self.__weights)
        print(f'EAP: {EAP}') 
        return EAP

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
    def ndata(self):
        return self.__ndata
    @property
    def data(self):
        return self.__data
    @property
    def para(self):
        return self.__para
    @property
    def weights(self):
        try:
            return self.__weights.ravel()
        except:
            raise ValueError(f'Did you add the observation data and conditioning?')
    @property
    def k_obs(self):
        return self.__k_obs

def test(data):
    import numpy as np
    import scipy as scp
    import matplotlib.pyplot as plt
    KBC=KernelABC(data=data.regression_data['ysim'],
                  para=data.regression_data['para'],
                  kernel=GaussKernel(sigma=1.))
    KBC.conditioning(obs=data.regression_data['yobs'])
    PostKernelMean = KBC.posterior_kernel_mean(GaussKernel(sigma=0.08))
    
    t = np.linspace(-3,3,200)
    mu_post =  np.dot(data.regression_data['xobs'],data.regression_data['yobs'].T) \
            /  (np.dot(data.regression_data['xobs'],data.regression_data['xobs'].T)+1)
    sigma_post = np.sqrt(1./(np.dot(data.regression_data['xobs'],data.regression_data['xobs'].T)+1))
    p_post = lambda x: scp.stats.norm.pdf(x,mu_post.ravel(),sigma_post.ravel())
    kde_val = PostKernelMean.kde(np.array([t]).T,weights_normalize=True,normalize=True)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(t,p_post(t),label='posterior')
    ax1.plot(t,kde_val,label='pdf estimation by KernelMean')
    plt.legend()
    fig.savefig('./pic/test/kernelabc/1dconditioning.png')
    
if __name__ == '__main__':
    from kernelmtd import data
    import os
    if not os.path.exists('./pic/test/kernelacb/'):
        os.makedirs('./pic/test/kernelabc/')
    test(data)
    