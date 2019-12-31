from sklearn.metrics.pairwise import *

import numpy as np
from sklearn.utils import check_array, gen_batches
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import issparse


def row_mahala_norms(X,Q,squared=False):
    """
    Row-wise (squared) Mahalanobis norm of X.
    Parameters
    ----------
    X : array_like
        The input array. X.shape should be (n_data, n_features)
    Q : array_like
        The input array. The shape should be (n_features,n_features).
        Q is symmetric matrix
    squared : bool, optional (default = False)
        If True, return squared norms.
    Returns
    -------
    array_like
        The row-wise (squared) Euclidean norm of X.
    """
    QXt = np.dot(Q,X.T)
    norms = np.einsum('ij,ji->i', X, QXt)
    if not squared:
        np.sqrt(norms, norms)
    return norms

def mahalanobis_distances(X, Y=None, Q=None, Y_norm_squared=None, squared=False, X_norm_squared=None):
    if Q is None:
        Q = np.eye(X.shape[1])
    
    X, Y = check_pairwise_arrays(X, Y)
    
    if X_norm_squared is not None:
        XX = check_array(X_norm_squared)
        if XX.shape == (1, X.shape[0]):
            XX = XX.T
        elif XX.shape != (X.shape[0], 1):
            raise ValueError(
                "Incompatible dimensions for X and X_norm_squared")
        if XX.dtype == np.float32:
            XX = None
    elif X.dtype == np.float32:
        XX = None
    else:
        XX = row_mahala_norms(X, Q, squared=True)[:, np.newaxis]

    if X is Y and XX is not None:
        # shortcut in the common case distances(X, X)
        YY = XX.T
    elif Y_norm_squared is not None:
        YY = np.atleast_2d(Y_norm_squared)

        if YY.shape != (1, Y.shape[0]):
            raise ValueError(
                "Incompatible dimensions for Y and Y_norm_squared")
        if YY.dtype == np.float32:
            YY = None
    elif Y.dtype == np.float32:
        YY = None
    else:
        YY = row_mahala_norms(Y, Q, squared=True)[np.newaxis, :]

    if X.dtype == np.float32:
        # matrix on chunks of X and Y upcast to float64
        distances = _mahalanobis_distances_upcast(X, XX, Y, YY, Q)
    else:
        # if dtype is already float64, no need to chunk and upcast
        distances = - 2 * safe_sparse_dot(X, np.dot(Q,Y.T), dense_output=True)
        distances += XX
        distances += YY
    np.maximum(distances, 0, out=distances)

    # Ensure that distances between vectors and themselves are set to 0.0.
    # This may not be the case due to floating point rounding errors.
    if X is Y:
        np.fill_diagonal(distances, 0)

    return distances if squared else np.sqrt(distances, out=distances)

def _mahalanobis_distances_upcast(X, XX=None, Y=None, YY=None, Q=None,batch_size=None):
    """
    Mahalanobis distances between X and Y
    Assumes X and Y have float32 dtype.
    Assumes XX and YY have float64 dtype or are None.
    X and Y are upcast to float64 by chunks.
    """
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    n_features = X.shape[1]
    
    if Q is None:
        Q = np.eye(n_features)
    
    distances = np.empty((n_samples_X, n_samples_Y), dtype=np.float32)

    if batch_size is None:
        x_density = X.nnz / np.prod(X.shape) if issparse(X) else 1
        y_density = Y.nnz / np.prod(Y.shape) if issparse(Y) else 1

        maxmem = max(
            ((x_density * n_samples_X + y_density * n_samples_Y) * n_features
             + (x_density * n_samples_X * y_density * n_samples_Y)) / 10,
            10 * 2 ** 17)

        # The increase amount of memory in 8-byte blocks is:
        # - x_density * batch_size * n_features (copy of chunk of X)
        # - y_density * batch_size * n_features (copy of chunk of Y)
        # - batch_size * batch_size (chunk of distance matrix)
        # Hence x² + (xd+yd)kx = M, where x=batch_size, k=n_features, M=maxmem
        #                                 xd=x_density and yd=y_density
        tmp = (x_density + y_density) * n_features
        batch_size = (-tmp + np.sqrt(tmp ** 2 + 4 * maxmem)) / 2
        batch_size = max(int(batch_size), 1)

    x_batches = gen_batches(n_samples_X, batch_size)

    for i, x_slice in enumerate(x_batches):
        X_chunk = X[x_slice].astype(np.float64)
        if XX is None:
            XX_chunk = row_mahala_norms(X_chunk, Q, squared=True)[:, np.newaxis]
        else:
            XX_chunk = XX[x_slice]

        y_batches = gen_batches(n_samples_Y, batch_size)

        for j, y_slice in enumerate(y_batches):
            Y_chunk = Y[y_slice].astype(np.float64)
            if YY is None:
                YY_chunk = row_mahala_norms(Y_chunk, Q, squared=True)[np.newaxis, :]
            else:
                YY_chunk = YY[:, y_slice]

            d = -2 * safe_sparse_dot(X_chunk, np.dot(Q,Y_chunk.T), dense_output=True)
            d += XX_chunk
            d += YY_chunk

            distances[x_slice, y_slice] = d.astype(np.float32, copy=False)

    return distances