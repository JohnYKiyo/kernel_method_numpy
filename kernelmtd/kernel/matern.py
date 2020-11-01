from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
import jax.scipy as scp
import scipy

import warnings

from ..metrics.distance import euclid_distance
from ..utils import pairwise, gradpairwise


def K_0p5(x1, x2, l):  # noqa: E741
    return np.exp(-euclid_distance(x1, x2, False) / l)


# K_0p5_pairwise = pairwise(pairwise(K_0p5,1),1) #(i,j,d),(k,l,d) -> (i,k,j,l)
# grad_K_0p5_pairwise = pairwise(gradpairwise(K_0p5,1),1) #(i,j,d),(k,l,d) -> (i,k,j,l,d)
K_0p5_pairwise = pairwise(K_0p5, 1)  # (i,d),(j,d) -> (i,j)
grad_K_0p5_pairwise = gradpairwise(K_0p5, 1)  # (i,d),(j,d) -> (i,j,d)


def K_1p5(x1, x2, l):  # noqa: E741
    K = euclid_distance(x1, x2, False) / l * np.sqrt(3)
    return (1. + K) * np.exp(-K)


# K_1p5_pairwise = pairwise(pairwise(K_1p5,1),1) #(i,j,d),(k,l,d) -> (i,k,j,l)
# grad_K_1p5_pairwise = pairwise(gradpairwise(K_1p5,1),1) #(i,j,d),(k,l,d) -> (i,k,j,l,d)
K_1p5_pairwise = pairwise(K_1p5, 1)  # (i,d),(j,d) -> (i,j)
grad_K_1p5_pairwise = gradpairwise(K_1p5, 1)  # (i,d),(j,d) -> (i,j,d)


def K_2p5(x1, x2, l):  # noqa: E741
    K = euclid_distance(x1, x2, False) / l * np.sqrt(5)
    return (1. + K + K ** 2 / 3.0) * np.exp(-K)


# K_2p5_pairwise = pairwise(pairwise(K_2p5,1),1) #(i,j,d),(k,l,d) -> (i,k,j,l)
# grad_K_2p5_pairwise = pairwise(gradpairwise(K_2p5,1),1) #(i,j,d),(k,l,d) -> (i,k,j,l,d)
K_2p5_pairwise = pairwise(K_2p5, 1)  # (i,d),(j,d) -> (i,j)
grad_K_2p5_pairwise = gradpairwise(K_2p5, 1)  # (i,d),(j,d) -> (i,j,d)


def K_inf(x1, x2, l):  # noqa: E741
    return np.exp(-euclid_distance(x1, x2, True) / 2.0 / l**2)


# K_inf_pairwise = pairwise(pairwise(K_inf,1),1) #(i,j,d),(k,l,d) -> (i,k,j,l)
# grad_K_inf_pairwise = pairwise(gradpairwise(K_inf,1),1) #(i,j,d),(k,l,d) -> (i,k,j,l,d)
K_inf_pairwise = pairwise(K_inf, 1)  # (i,d),(j,d) -> (i,j)
grad_K_inf_pairwise = gradpairwise(K_inf, 1)  # (i,d),(j,d) -> (i,j,d)


def K_other_pairwise(x1, x2, l, nu):  # noqa: E741
    dists = pairwise(euclid_distance, 1)
    dists_matrix = dists(x1, x2, False) / l
    dists_matrix = np.where(dists_matrix == 0, np.finfo(float).eps, dists_matrix)
    tmp = (np.sqrt(2 * nu) * dists_matrix)
    val = (2 ** (1. - nu)) / np.exp(scp.special.gammaln(nu))
    return val * tmp**nu * scipy.special.kv(nu, tmp)


def grad_K_ohter_pairwise(x1, x2, l, nu):  # noqa: E741
    raise NotImplementedError


def matern(x, y, l=1., nu=1.5):  # noqa: E741
    if nu == 0.5:
        return K_0p5_pairwise(x, y, l)

    elif nu == 1.5:
        return K_1p5_pairwise(x, y, l)

    elif nu == 2.5:
        return K_2p5_pairwise(x, y, l)

    elif nu == np.inf:
        return K_inf_pairwise(x, y, l)

    else:
        warnings.warn('Slow processing speed. cuz not use XLA compiler.', Warning)
        return K_other_pairwise(x, y, l, nu)


def grad_matern(x, y, l=1., nu=0.5):  # noqa: E741
    if nu == 0.5:
        return grad_K_0p5_pairwise(x, y, l)
    elif nu == 1.5:
        return grad_K_1p5_pairwise(x, y, l)

    elif nu == 2.5:
        return grad_K_2p5_pairwise(x, y, l)

    elif nu == np.inf:
        return grad_K_inf_pairwise(x, y, l)
    else:
        warnings.warn('Slow processing speed. cuz not use XLA compiler.', Warning)
        return grad_K_ohter_pairwise(x, y, l, nu)


class MaternKernel(object):
    """Matern kernel.
    The class of Matern kernels.
    It has a :math:`\\nu` parameter which controls the smoothness of the function.
    When :math:`\\nu` is 0.5, the kernel becomes Laplace kernel (absolute exponential kernel).
    As :math:`\\nu\\rightarrow\\infty`, the kernel becomes equivalent to Gaussian kernel.

    Args:
        a (float > 0, optional): Multiply the output of the function by this value. Defaults to 1.0.
        l (float > 0, optional): The length scale of the kernel. Defaults to 1.0.
        nu (float > 0, optional): Defaults to 1.5.
            The parameter nu controlling the smoothness of function. The smaller nu, the less smooth function is.
            For nu=inf, the kernel becomes equivalent to the Gaussian kernel.
            For nu=0.5 the kernel becomes Laplace kernel.

    Note:
        If you use a value other than the following nu, the calculation time will be significantly worse, due to not use XLA compiling.
        nu = [0.5,1.5,2.5,np.inf]

    Examples:
        >>> import numpy as onp
        >>> kernel_matern = MaternKernel(l=1.,nu=0.5)
        >>> x = np.atleast_2d(np.linspace(-5.,5.,3)).T
        >>> onp.asarray(kernel_matern.kde(x,np.array([[0.],[1.]])))
        array([[0.00673795, 0.00247875],
               [1.        , 0.36787944],
               [0.00673795, 0.01831564]])
    """

    def __init__(self, a=1.0, l=1.0, nu=1.5, *args, **kwargs):  # noqa: E741
        self.__a = a
        self.__l = l
        self.__nu = nu

    def __call__(self, x1, x2, **kwargs):
        """compute kernel density

        Args:
            x1 (ndarray): ndarray of shape (n_samples_x1, n_dim).
            x2 (ndarray): ndarray of shape (n_samples_x2, n_dim).

        Returns:
            KV (ndarray): return kernel value tensor. ndarray of shape (n_samples_x1,n_samples_x2).
                Kernel k(x1,x2)
        """
        return self.kde(x1, x2)

    def kde(self, x1, x2, **kwargs):
        """compute kernel density

        Args:
            x1 (ndarray): ndarray of shape (n_samples_x1, n_dim).
            x2 (ndarray): ndarray of shape (n_samples_x2, n_dim).
            **kwargs : Not specified.

        Returns:
            KV (ndarray): return kernel value tensor. ndarray of shape (n_samples_x1,n_samples_x2).
                Kernel k(x1,x2)
        """
        return self.__a * matern(x1, x2, self.__l, self.__nu)

    def logkde(self, x1, x2, **kwargs):
        """compute kernel density

        Args:
            x1 (ndarray): ndarray of shape (n_samples_x1, n_dim).
            x2 (ndarray): ndarray of shape (n_samples_x2, n_dim).
            **kwargs : Not specified.

        Returns:
            KV (ndarray): return kernel value tensor. ndarray of shape (n_samples_x1,n_samples_x2).
                Kernel log(k(x1,x2))
        """
        return np.log(self.__a * matern(x1, x2, self.__l, self.__nu))

    def gradkde(self, x1, x2, **kwargs):
        """compute gradient of kernel density

        Args:
            x1 (ndarray): ndarray of shape (n_samples_x1, n_dim).
            x2 (ndarray): ndarray of shape (n_samples_x2, n_dim).
            **kwargs : Not specified.

        Returns:
            KV (ndarray): return gradient value tensor. ndarray of shape (n_samples_x1,n_samples_x2, n_dim).
                Derivative value at x1 of kernel centered on x2. dk(x,x2)/dx (x=x1)
        """
        return self.__a * grad_matern(x1, x2, self.__l, self.__nu)

    @property
    def a(self):
        return self.__a

    @property
    def l(self):  # noqa: E741 E743
        return self.__l

    @property
    def nu(self):
        return self.__nu


if __name__ == "__main__":
    import doctest
    doctest.testmod()
