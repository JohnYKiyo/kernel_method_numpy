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
    
    test(testdata)
    