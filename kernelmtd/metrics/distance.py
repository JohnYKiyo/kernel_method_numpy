from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np
from jax import jit

@jit
def euclid_distance(x,y, square=True):
    '''
    d(x,y) = (x-y)^2
    '''
    XX=np.dot(x,x)
    YY=np.dot(y,y)
    XY=np.dot(x,y)
    if not square:
        return np.sqrt(XX+YY-2*XY)
    return XX+YY-2*XY

@jit
def mahalanobis_distance(x,y,Q,square=True):
    '''
    d(x,y) = (x-y)Q(x-y)
    '''
    XQX=np.dot(x,np.dot(Q,x))
    YQY=np.dot(y,np.dot(Q,y))
    XQY=np.dot(x,np.dot(Q,y))
    YQX=np.dot(y,np.dot(Q,x))
    if not square:
        return np.sqrt(XQX+YQY-XQY-YQX)
    return XQX+YQY-XQY-YQX
