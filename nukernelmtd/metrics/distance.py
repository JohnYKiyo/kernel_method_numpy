import numpy as np


def pairwise_euclid_distances(x, y, square=True):
    """
    return a pairwise distance matrix.
    d_ij = (x_i-y_i)^2
    Args:
        x (2d numpy.array): [description]
        y (2d numpy.array): [description]
        square (bool, optional):
            Set to True to square the return value, (x-y)^2,
            if set to False, return |x-y|. Defaults to True.

    Returns:
        array: if square is Ture, return (x-y)^2, else, return |x-y|.

    Examples:
        >>> pairwise_euclid_distances(np.array([[1.,2.]]),np.array([[3.,4.]]))
        array([[8.]])
        >>> pairwise_euclid_distances(np.array([[1.,2.],[3.,4.]]),np.array([[1.,1.],[1.,2.]]))
        array([[ 1.,  0.],
               [13.,  8.]])
    """
    XX = np.einsum('id,id->i', x, x)[:, np.newaxis]
    YY = np.einsum('id,id->i', y, y)[np.newaxis, :]
    XY = np.einsum('id,jd->ij', x, y)
    if not square:
        return np.sqrt(XX + YY - 2. * XY)
    return XX + YY - 2. * XY


def pairwise_mahalanobis_distances(x, y, Q, square=True):
    """[summary]
    d(x,y) = (x-y)Q(x-y)
    Args:
        x (2d numpy.array): [description]
        y (2d numpy.array): [description]
        square (bool, optional):
            Set to True to square the return value, (x-y)Q(x-y)^T,
            if set to False, return ((x-y)Q(x-y)^T)^0.5. Defaults to True.

    Returns:
        array: if square is Ture, return (x-y)Q(x-y)^T, else, return ((x-y)Q(x-y)^T)^0.5.

    Examples:
        >>> pairwise_mahalanobis_distances(np.array([[1,2]]),np.array([[3,4.]]),np.array([[0.5,1],[4,0.5]]))
        array([[24.]])

        >>> pairwise_mahalanobis_distances(np.array([[1,2]]),np.array([[3,4.]]),np.array([[0.5,1],[4,0.5]]),False)
        array([[4.89897949]])

        >>> pairwise_mahalanobis_distances(np.array([[1.,2.],[3.,4.]]),np.array([[1.,1.],[1.,2.]]),np.array([[1.,2.],[1.,1.]]))
        array([[ 1.,  0.],
               [31., 20.]])
    """
    XQX = np.einsum('ij,jk,ik->i', x, Q, x)[:, np.newaxis]
    YQY = np.einsum('ij,jk,ik->i', y, Q, y)[np.newaxis, :]
    XQY = np.einsum('ij,jk,lk->il', x, Q, y)
    YQX = np.einsum('ij,jk,lk->li', y, Q, x)
    if not square:
        return np.sqrt(XQX + YQY - XQY - YQX)
    return XQX + YQY - XQY - YQX


if __name__ == '__main__':
    import doctest
    doctest.testmod()
