from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np


def euclid_distance(x, y, square=True):
    """[summary]
    d(x,y) = (x-y)^2
    Args:
        x (1d numpy.array): [description]
        y (1d numpy.array): [description]
        square (bool, optional):
            Set to True to square the return value, (x-y)^2,
            if set to False, return |x-y|. Defaults to True.

    Returns:
        array: if square is Ture, return (x-y)^2, else, return |x-y|.

    Examples:
        >>> euclid_distance(np.array([1.,2.]),np.array([3.,4.])).item()
        8.0

        >>> euclid_distance(np.array([1.,2.]),np.array([3.,4.]),False).item()
        2.8284271247461903

        >>> euclid_distance_jit = jit(euclid_distance,static_argnums=(2,))
        >>> euclid_distance_jit(np.array([1.,2.]),np.array([3.,4.]),True).item()
        8.0

        >>> euclid_distance_jit = jit(euclid_distance,static_argnums=(2,))
        >>> euclid_distance_jit(np.array([1.,2.]),np.array([3.,4.]),False).item()
        2.8284271247461903

    """

    XX = np.dot(x, x)
    YY = np.dot(y, y)
    XY = np.dot(x, y)
    if not square:
        return np.sqrt(XX + YY - 2. * XY)
    return XX + YY - 2. * XY


def mahalanobis_distance(x, y, Q, square=True):
    """[summary]
    d(x,y) = (x-y)Q(x-y)
    Args:
        x (1d numpy.array): [description]
        y (1d numpy.array): [description]
        square (bool, optional):
            Set to True to square the return value, (x-y)Q(x-y)^T,
            if set to False, return ((x-y)Q(x-y)^T)^0.5. Defaults to True.

    Returns:
        array: if square is Ture, return (x-y)Q(x-y)^T, else, return ((x-y)Q(x-y)^T)^0.5.

    Examples:
        >>> mahalanobis_distance(np.array([1,2]),np.array([3,4.]),np.array([[0.5,1],[4,0.5]])).item()
        24.0

        >>> mahalanobis_distance(np.array([1,2]),np.array([3,4.]),np.array([[0.5,1],[4,0.5]]),False).item()
        4.898979485566356

        >>> mahalanobis_distance_jit = jit(mahalanobis_distance,static_argnums=(3,))
        >>> mahalanobis_distance_jit(np.array([1,2]),np.array([3,4.]),np.array([[0.5,1],[4,0.5]]),True).item()
        24.0

        >>> mahalanobis_distance_jit = jit(mahalanobis_distance,static_argnums=(3,))
        >>> mahalanobis_distance_jit(np.array([1,2]),np.array([3,4.]),np.array([[0.5,1],[4,0.5]]),False).item()
        4.898979485566356

    """
    XQX = np.dot(x, np.dot(Q, x))
    YQY = np.dot(y, np.dot(Q, y))
    XQY = np.dot(x, np.dot(Q, y))
    YQX = np.dot(y, np.dot(Q, x))
    if not square:
        return np.sqrt(XQX + YQY - XQY - YQX)
    return XQX + YQY - XQY - YQX


if __name__ == '__main__':
    import doctest
    doctest.testmod()
