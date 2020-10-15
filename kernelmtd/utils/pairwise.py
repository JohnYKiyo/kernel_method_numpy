from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import jit, vmap,grad
from functools import partial

def pairwise(func,**kwargs):
    """pairwisation
    Args:
        func (function):
            for automatic vectorization
            d_ij = func(X_i , Y_j).
            i,j are assumed to indicate the data index.

    Returns:
        function : the function that returns d_ij array.

    Examples:
        >>> import sys
        >>> sys.path.append('../')
        >>> from metrics.distance import mahalanobis_distance,euclid_distance
        >>> test = pairwise(euclid_distance,square=True)
        >>> test(np.array([[1.,2.],[3,4],[5.,6.,]]),np.array([[1.,2.],[2.,1.],[3.,3.]]))
        DeviceArray([[ 0.,  2.,  5.],
                     [ 8., 10.,  1.],
                     [32., 34., 13.]], dtype=float64)

        >>> import sys
        >>> sys.path.append('../')
        >>> from metrics.distance import mahalanobis_distance,euclid_distance
        >>> test = pairwise(mahalanobis_distance,Q=np.array([[4.,1.],[2.,2.]]),square=True)
        >>> test(np.array([[1.,2.],[3,4],[5.,6.,]]),np.array([[1.,2.],[2.,1.],[3.,3.]]))
        DeviceArray([[  0.,   3.,  24.],
                     [ 36.,  31.,   2.],
                     [144., 131.,  52.]], dtype=float64)

    """
    return jit(vmap(vmap(partial(func,**kwargs),in_axes=(None,0)),in_axes=(0,None)))

def gradpairwise(func,**kwargs):
    """pairwisation
    Args:
        func (function):
            for automatic vectorization
            d_ij = dfunc/dX (X_i,Y_j)
            i,j are assumed to indicate the data index.

    Returns:
        function : [description]
    
    Examples:
        >>> import sys
        >>> sys.path.append('../')
        >>> from metrics.distance import mahalanobis_distance,euclid_distance
        >>> test = gradpairwise(euclid_distance,square=True)
        >>> test(np.array([[1.,2.],[3,4],[5.,6.,]]),np.array([[1.,2.],[2.,1.],[3.,3.]]))
        DeviceArray([[[ 0.,  0.],
                      [-2.,  2.],
                      [-4., -2.]],
        <BLANKLINE> 
                     [[ 4.,  4.],
                      [ 2.,  6.],
                      [ 0.,  2.]],
        <BLANKLINE> 
                     [[ 8.,  8.],
                      [ 6., 10.],
                      [ 4.,  6.]]], dtype=float64)
    
    """
    return jit(vmap(vmap(grad(partial(func,**kwargs)),in_axes=(None,0)),in_axes=(0,None)))

if __name__ == '__main__':
    import doctest
    doctest.testmod()