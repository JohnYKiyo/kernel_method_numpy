from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import jit, vmap,grad

def pairwise(function,Nopts=0):
    """return vectorized function
    Args:
        function (function):
            This pairwise function automatic vectorize input function when defined as scalar-output.
            d_ij = function(X_i , Y_j).
            i,j are assumed to indicate the data index.
        
        Nopts (int, optional): function. Defaults to 0.
            Number of option arguments that the function has
    Returns:
        function : the function that returns d_ij array.
    
    Examples:
        >>> import sys
        >>> sys.path.append('../')
        >>> from metrics.distance import mahalanobis_distance,euclid_distance
        >>> test = pairwise(euclid_distance,1)
        >>> test(np.array([[1.,2.],[3.,4.],[5.,6.,]]),np.array([[1.,2.],[2.,1.],[3.,3.]]),True)
        DeviceArray([[ 0.,  2.,  5.],
                     [ 8., 10.,  1.],
                     [32., 34., 13.]], dtype=float64)

        >>> import sys
        >>> sys.path.append('../')
        >>> from metrics.distance import mahalanobis_distance,euclid_distance
        >>> test = pairwise(mahalanobis_distance,2)
        >>> test(np.array([[1.,2.],[3,4],[5.,6.,]]),np.array([[1.,2.],[2.,1.],[3.,3.]]),np.array([[4.,1.],[2.,2.]]),True)
        DeviceArray([[  0.,   3.,  24.],
                     [ 36.,  31.,   2.],
                     [144., 131.,  52.]], dtype=float64)
    
    """    
    add_in_axes= [None]*Nopts
    static_argnums=[2+i for i in range(Nopts)]
    return jit(
        vmap(
            vmap(function,in_axes=(None,0, *add_in_axes)),
            in_axes=(0,None, *add_in_axes)),
        static_argnums=static_argnums)

def gradpairwise(function,Nopts=0):
    """return vectorized gradient function
    Args:
        function (function):
            This gradpairwise function automatic gradiented vectorize input function when defined as scalar-output.
            d_ij = d function(X_i , Y_j) / dX .
            i,j are assumed to indicate the data index.
        
        Nopts (int, optional): function. Defaults to 0.
            Number of option arguments that the function has
    Returns:
        function : the function that returns d_ij array.
    
    Examples:
        >>> import sys
        >>> sys.path.append('../')
        >>> from metrics.distance import mahalanobis_distance,euclid_distance
        >>> test = gradpairwise(euclid_distance,1)
        >>> test(np.array([[1.,2.],[3,4],[5.,6.,]]),np.array([[1.,2.],[2.,1.],[3.,3.]]),True)
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
    add_in_axes= [None]*Nopts
    static_argnums=[2+i for i in range(Nopts)]
    return jit(
        vmap(
            vmap(grad(function),in_axes=(None,0, *add_in_axes)),
            in_axes=(0,None, *add_in_axes)),
        static_argnums=static_argnums)
        
if __name__ == '__main__':
    import doctest
    doctest.testmod()