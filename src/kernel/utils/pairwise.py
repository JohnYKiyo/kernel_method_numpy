from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import jit, vmap
from functools import partial

def pairwise(func,**kwargs):
    """pairwisation
    Args:
        func (function):
            for automatic vectorization
            d_ij = func(X_i , Y_j).
            i,j are assumed to indicate the data index.

    Returns:
        function : [description]
    """
    return jit(vmap(vmap(partial(func,**kwargs),in_axes=(None,0)),in_axes=(0,None)))

def gradpairwise(func,**kwargs):
    """pairwisation
    Args:
        func (function):
            for automatic vectorization
            d_ij = d(func(X_i , Y_j)/dX.
            i,j are assumed to indicate the data index.

    Returns:
        function : [description]
    """
    return jit(vmap(vmap(grad(partial(func,**kwargs),in_axes=(None,0)),in_axes=(0,None))))