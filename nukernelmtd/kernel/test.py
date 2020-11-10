from .matern import K_0p5_pairwise, grad_K_0p5_pairwise
from .matern import K_1p5_pairwise, grad_K_1p5_pairwise
from .matern import K_2p5_pairwise, grad_K_2p5_pairwise
from .matern import K_inf_pairwise, grad_K_inf_pairwise
from .matern import K_other_pairwise
from .matern import MaternKernel
from .gaussian_rbf import gauss1d_pairwise, grad_gauss1d_pairwise, gauss_pairwise, grad_gauss_pairwise, GaussKernel

import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn.gaussian_process.kernels import Matern as sk_Matern
from sklearn.gaussian_process.kernels import RBF as sk_Gauss

import warnings


def test_matern_nu(nu=0.5):
    if nu == 0.5:
        string = '0p5'
        kernel = K_0p5_pairwise
        grad = grad_K_0p5_pairwise
    elif nu == 1.5:
        string = '1p5'
        kernel = K_1p5_pairwise
        grad = grad_K_1p5_pairwise
    elif nu == 2.5:
        string = '2p5'
        kernel = K_2p5_pairwise
        grad = grad_K_2p5_pairwise
    elif nu == np.inf:
        string = 'infty'
        kernel = K_inf_pairwise
        grad = grad_K_inf_pairwise
    else:
        string = f'{nu}'
        warnings.warn('Slow processing speed. cuz not use XLA compiler.', Warning)
        kernel = K_other_pairwise

    x = np.linspace(-10, 10, 1000).reshape(1000, 1)
    try:
        plt.plot(x, np.squeeze(kernel(x, np.array([[0.]]), 1.)), label=f'Matern nu={nu}')
    except:  # noqa
        pass
    try:
        plt.plot(x, np.squeeze(grad(x, np.array([[0.]]), 1.)), label='gradient')
    except:  # noqa
        pass
    plt.legend()
    plt.savefig(f'./pic/test/kernel/matern_nu_{string}.png')
    plt.close()


def check_matern_sklearn(nu):
    x = np.atleast_2d(np.linspace(-10., 10., 100)).T
    skmatern = sk_Matern(length_scale=1., nu=nu)
    return skmatern(x, np.array([[0], [1]]))


def check_matern_kernelmtd(nu):
    x = np.atleast_2d(np.linspace(-10., 10., 100)).T
    matk = MaternKernel(l=1., nu=nu)
    return matk.kde(x.reshape(100, 1), np.array([[0], [1]]))


def test_matern_equivalent_function():
    print('test nu=0.5')
    np.testing.assert_almost_equal(check_matern_sklearn(0.5),
                                   check_matern_kernelmtd(0.5))
    print(f'nu=0.5 sklearn:{timeit.timeit("check_matern_sklearn(0.5)", globals = globals(), number=1000)/1000:.8f}')
    print(f'nu=0.5 kernelmtd:{timeit.timeit("check_matern_kernelmtd(0.5)", globals = globals(), number=1000)/1000:.8f}')

    print('test nu=1.5')
    np.testing.assert_almost_equal(check_matern_sklearn(1.5),
                                   check_matern_kernelmtd(1.5))
    print(f'nu=1.5 sklearn:{timeit.timeit("check_matern_sklearn(1.5)", globals = globals(), number=1000)/1000:.8f}')
    print(f'nu=1.5 kernelmtd:{timeit.timeit("check_matern_kernelmtd(1.5)", globals = globals(), number=1000)/1000:.8f}')

    print('test nu=2.5')
    np.testing.assert_almost_equal(check_matern_sklearn(2.5),
                                   check_matern_kernelmtd(2.5))
    print(f'nu=2.5 sklearn:{timeit.timeit("check_matern_sklearn(2.5)", globals = globals(), number=1000)/1000:.8f}')
    print(f'nu=2.5 kernelmtd:{timeit.timeit("check_matern_kernelmtd(2.5)", globals = globals(), number=1000)/1000:.8f}')

    print('test nu=np.inf')
    np.testing.assert_almost_equal(check_matern_sklearn(np.inf),
                                   check_matern_kernelmtd(np.inf))
    print(f'nu=inf sklearn:{timeit.timeit("check_matern_sklearn(np.inf)", globals = globals(), number=1000)/1000:.8f}')
    print(f'nu=inf kernelmtd:{timeit.timeit("check_matern_kernelmtd(np.inf)", globals = globals(), number=1000)/1000:.8f}')


def test_gauss_1d():
    x = np.linspace(-10, 10, 100).reshape(100, 1)
    plt.plot(x, np.squeeze(gauss1d_pairwise(x, np.array([[0.]]), 1.)), label='gauss1d')
    plt.plot(x, np.squeeze(grad_gauss1d_pairwise(x, np.array([[0.]]), 1.)), label='gradient')
    plt.legend()
    plt.savefig('./pic/test/kernel/gauss1d.png')
    plt.close()


def test_gauss():
    XX, YY = np.meshgrid(np.arange(-2, 2, 0.1), np.arange(-2, 2, 0.1))
    S = np.array([[1., 0.5], [0.5, 1.]])
    Q = np.linalg.inv(S)
    kde = gauss_pairwise(np.c_[XX.ravel(), YY.ravel()],
                         np.array([[0., 0.]]),
                         Q)
    gradkde = grad_gauss_pairwise(np.c_[XX.ravel(), YY.ravel()],
                                  np.array([[0., 0.]]),
                                  Q)
    c = np.sqrt(gradkde[:, 0, 0]**2 + gradkde[:, 0, 1]**2)
    plt.contour(XX, YY, kde.reshape(XX.shape[0], -1))
    plt.quiver(XX.ravel(), YY.ravel(), gradkde[:, 0, 0], gradkde[:, 0, 1], c)
    plt.savefig('./pic/test/kernel/gauss.png')
    plt.close()


def check_gauss_sklearn():
    x = np.atleast_2d(np.linspace(-10., 10., 100)).T
    skgauss = sk_Gauss(length_scale=1.)
    return skgauss(x, np.array([[0.], [-2.5], [5.]]))


def check_gauss_kernelmtd():
    x = np.atleast_2d(np.linspace(-10., 10., 100)).T
    gk = GaussKernel(sigma=1.)
    return gk.kde(x, np.array([[0.], [-2.5], [5.]]))


def test_gauss_equivalent_function():
    print('test sigma=1.')
    np.testing.assert_almost_equal(check_gauss_sklearn(),
                                   check_gauss_kernelmtd())
    print(f'sklearn:{timeit.timeit("check_gauss_sklearn()", globals = globals(), number=1000)/1000:.8f}')
    print(f'kernelmtd:{timeit.timeit("check_gauss_kernelmtd()", globals = globals(), number=1000)/1000:.8f}')


def main():
    test_matern_nu(0.5)
    test_matern_nu(1.5)
    test_matern_nu(2.5)
    test_matern_nu(np.inf)
    test_matern_equivalent_function()
    test_gauss_1d()
    test_gauss()
    test_gauss_equivalent_function()


if __name__ == '__main__':
    import os
    if not os.path.exists('./pic/test/kernel'):
        os.makedirs('./pic/test/kernel')
    main()
