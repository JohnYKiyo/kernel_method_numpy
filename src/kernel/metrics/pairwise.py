from jax.config import config; config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from jax import jit, vmap
from functools import partial

def euclid_distance(x,y, square=True):
    '''
    \sum_m (X_m - Y_m)^2
    '''
    XX=np.dot(x,x)
    YY=np.dot(y,y)
    XY=np.dot(x,y)
    if not square:
        return np.sqrt(XX+YY-2*XY)
    return XX+YY-2*XY

def pairwise_distances(dist,**arg):
    '''
    d_ij = dist(X_i , Y_j)
    "i,j" are assumed to indicate the data index.
    '''
    return jit(vmap(vmap(partial(dist,**arg),in_axes=(None,0)),in_axes=(0,None)))