from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
from functools import partial
import numpy as onp
from scipy.optimize import minimize
from scipy import random
from tqdm import tqdm 
import warnings

from ..kernel_mean import KernelMean
from ..kernel import GaussKernel,MaternKernel

class KernelHerding(object):
    def __init__(self, KernelMeanObj):
        if not isinstance(KernelMeanObj, KernelMean):
            raise TypeError(f"init object shuld be KernelMean class, but got {type(KernelMeanObj)}")

        self.__KernelMean = KernelMeanObj
        self.__kernel = KernelMeanObj.kernel
        self.__samples = None
        
    def sampling(self, sample_size, **kwargs):
        """[summary]
        Args:
            sample_size (int): sample size of herding.
            **kwargs:
                see scipy.optimizer.minimize kwargs.
                
                normalize (bool, optional): Defaults to False. 
                    Specify `True` when normalizing the kernel. Some kernels do not have a normalization option, so see the kernel's docstrings.
                
                weights_normalize (bool, optional): Defaults to False.
                    Specify `True` when normalizing kernel average weighting.
                max_trial (int, optional): Defaults to 2.
                derivatives (bool, optional): Defaults to False.
                
        Returns:
            ndarray : supersamples.
        """
        normalize = kwargs.pop('normalize', False)
        weights_normalize = kwargs.pop('weights_normalize', False)
        max_trial = kwargs.pop('max_trial', 2)
        derivatives = kwargs.pop('derivatives',False)
        
        self.__samples = None
        h = partial(self.__KernelMean.kde, normalize=normalize, weights_normalize=weights_normalize)
        grad_h = None
        if derivatives:
            try:
                self.__KernelMean.gradkde(self.__KernelMean.data.iloc[0:1,:]) #check gradkde is defined.
                derivatives=True
                grad_h = partial(self.__KernelMean.gradkde, normalize=normalize, weights_normalize=weights_normalize)
            except:
                derivatives=False
                warnings.warn('kernel has no gradkde method. So, run with derivative option False', Warning)
        with tqdm(total=sample_size) as bar:
            x1,_ = self.__argmax(h, grad_h, derivatives=derivatives, max_trial=max_trial, **kwargs)
            samples = np.atleast_2d(x1)
            bar.update()
            for idx in range(sample_size-1):
                h,h_prime = self.__herding_update(samples, normalize=normalize, weights_normalize=weights_normalize, derivatives=derivatives, **kwargs)
                x,_ = self.__argmax(h, h_prime, derivatives=derivatives, max_trial=max_trial, **kwargs)
                samples = np.append(samples,np.atleast_2d(x),axis=0)
                self.__samples = samples
                
                post_str = f'param:{x}'
                bar.set_description_str(f'KernelHerding')
                bar.set_postfix_str(post_str)
                bar.update()
        return self.__samples

    def __herding_update(self, samples, normalize=False, weights_normalize=False, derivatives=False, **kwargs):
        f = lambda x: self.__KernelMean.kde(x, normalize=normalize, weights_normalize=weights_normalize) - np.mean(self.__kernel.kde(x, samples, normalize=normalize), axis=1, keepdims=True)
        
        f_prime = None
        if derivatives:
            f_prime = lambda x: self.__KernelMean.gradkde(x, weights_normalize=weights_normalize, normalize=normalize) - np.mean(self.__kernel.gradkde(x, samples,normalize=normalize),axis=1,keepdims=True)

        return f, f_prime
    
    def __argmax(self, h, h_prime, derivatives=False, max_trial=2, **kwargs):
        minus_h = lambda x: onp.array(np.squeeze(-1.*h(np.atleast_2d(x))))
        minus_h_prime = None
        if derivatives:
            minus_h_prime = lambda x: onp.array(np.squeeze(-1.*h_prime(np.atleast_2d(x))))
        x, val = self.__optimizer_scipy(minus_h, minus_h_prime, derivatives=derivatives, max_trial=max_trial, **kwargs)
        return x, val
    
    def __optimizer_scipy(self, h, h_prime, derivatives=False, max_trial=2, **kwargs):
        kwargs['method']='L-BFGS-B'
        if 'x0' not in kwargs:
            kwargs['x0'] = np.squeeze(self.__KernelMean.data.sample().values)
        if derivatives:
            kwargs['jac'] = h_prime
            
        x, val = kwargs['x0'], np.inf
        for i in range(max_trial):
            optimize_fail = True
            while optimize_fail:
                optimize_result = minimize(h, **kwargs)
                if not optimize_result.success:
                    kwargs['x0'] = random.uniform(low=np.min(self.__KernelMean.data,  axis=0),
                                                  high=np.max(self.__KernelMean.data, axis=0))
                    continue
                optimize_fail = False

            if optimize_result.fun < val:
                x = optimize_result.x
                val = optimize_result.fun
        return x, val

    @property
    def supersamples(self):
        if self.__samples is None:
            raise ValueError(f'super samples do not exist.')
        return self.__samples