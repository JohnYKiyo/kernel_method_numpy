import pandas as pd

from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
from jax import jit

from .utils import transform_data
from .kernel import GaussKernel, MaternKernel

class KernelMean(object):
    """[summary]
    The class of Kernel Mean.
    
    Args:
        data (array-like or pandas dataframe): 
            Sample for calculating kernel mean.
            An array of shape should be (n_samples_x1, n_dim).
            
        kernel (class kernel): 
            Specify an instance of kernel class as shown in Example.
        
        weights (array-like, optional): Defaults to None.
            Weight each sample when calculating the kernel mean. 
            By default, all samples are treated as equal weight.
            Conditional kernel mean can be obtained by specifying the weights conditioned by kernel ABC.
            
    Examples:
        >>> from kernelmtd.kernel import GaussKernel
        >>> import numpy as np 
        >>> kernel_gauss = GaussKernel(sigma=1.)
        >>> data = np.array([[0.],[1.],[3.]])
        >>> kernelmean = KernelMean(data=data,kernel=kernel_gauss,weights=None)
        >>> x = np.array([[-3],[-2.], [0.], [1.], [5.]])
        >>> np.array(kernelmean(x))
        array([[0.00381482],
               [0.048816  ],
               [0.53921322],
               [0.58062198],
               [0.04522482]])
    """    
    def __init__(self, data, kernel, weights=None): 
        if isinstance(data,pd.DataFrame):
            self._data = data
        else:
            self._data = pd.DataFrame(data)
        self._n_samples, self._n_features = self._data.shape
        self._weights = np.atleast_2d(np.full(self._n_samples,1./self._n_samples)).T
        if weights is not None:
            if not isinstance(weights,np.ndarray):
                raise ValueError(f'weights should be ndarray')
            if len(weights) != self._n_samples:
                raise ValueError(f'length of weights should be {self._n_samples,1}')
            self._weights = weights.reshape(self._n_samples,1)
        
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
   
    def __repr__(self):
        val =  f'Class: KernelMean\n'
        val += f'The number of data: {self.n_samples}\n'
        val += f'Kernel: \n{self.kernel}\n'
        return val
     
    def kde(self,val,**kwargs):
        """compute kernel density
        Args:
            val (ndarray): ndarray of shape (n_samples_val, n_dim).

        Returns:
            KV (ndarray): return gradient value tensor. ndarray of shape (n_samples_val,n_samples_data).
                Derivative value at val of kernel centered on data. dk(x,data)/dx (x=val)
        """        
        val = transform_data(val)
        kde = self.kernel.kde(val,self._data.values,**kwargs)
        #return np.average(kde,weights=self._weights,axis=1)
        return np.dot(kde,self._weights)
    
    def gradkde(self,val,**kwargs):
        """compute gradient of kernel density
        Args:
            val (ndarray): ndarray of shape (n_samples_val, n_dim).

        Returns:
            KV (ndarray): return gradient value tensor. ndarray of shape (n_samples_val,n_samples_data, n_dim).
                Derivative value at val of kernel centered on data. dk(x,data)/dx (x=val)

        Raises:
            NotImplementedError: If the gradient cannot be calculated, it returns a NotImplementedError.
        """        
        val = transform_data(val)
        try:
            grad = self.kernel.gradkde(val,self._data.values,**kwargs)
        except:
            raise NotImplementedError
        #return np.average(grad,weights=self._weights,axis=1)
        return np.einsum('ijd,jk->id',grad,self._weights)

    @property
    def kernel(self):
        return self.__kernel

    @property
    def weights(self):
        return self._weights
    
    @property
    def data(self):
        return self._data
    
    @property
    def n_samples(self):
        return self._n_samples
    
def test(data):
    plt.figure()
    plt.title('plot sample data')
    sns.scatterplot(x='0',y='1',data=data)
    plt.savefig('./pic/test/kernelmean/sample.png')
    plt.close()
    
    #bandwidth selection by scott
    bw = Bandwidth(data=data,method='scott',weights=None)
    print(bw)
    kernel = GaussKernel(sigma=bw.bandwidth)
    kernelmean = KernelMean(data=data,kernel=kernel,weights=None)
    
    x = np.arange(-10.0,10,0.5)
    y = np.arange(-10.0,10,0.5)
    xx,yy = np.meshgrid(x,y)
    z  = kernelmean(np.array([xx.ravel(),yy.ravel()]).T)
    z_grad = kernelmean.gradkde(np.array([xx.ravel(),yy.ravel()]).T)
    zz = z.reshape(xx.shape[0],yy.shape[0])
    c = np.mean(np.square(z_grad),axis=1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contour(xx,yy,zz)
    sns.scatterplot(x='0',y='1',data=data,ax=ax)
    plt.quiver(xx.ravel(),yy.ravel(),z_grad[:,0],z_grad[:,1],c)
    fig.savefig('./pic/test/kernelmean/kernelmean.png')
    
if __name__ == '__main__':
    from .data import testdata
    from .utils import Bandwidth
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    if not os.path.exists('./pic/test/kernelmean/'):
        os.makedirs('./pic/test/kernelmean/')
    import doctest
    
    test(testdata)
    doctest.testmod()
    