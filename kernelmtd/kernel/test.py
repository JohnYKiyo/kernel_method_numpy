import sys
sys.path.append('./')
from kernelmtd.kernel.matern import *
import os
if not os.path.exists('./pic/'):
    os.makedirs('./pic/')

import numpy as np
import matplotlib.pyplot as plt
import timeit
from sklearn.gaussian_process.kernels import Matern as sk_Matern
import jax.numpy as jnp

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
    
    
    
    x = np.linspace(-10,10,1000).reshape(1000,1)
    try:
        plt.plot(x,np.squeeze(kernel(x,np.array([[0.]]),1.)),label=f'Matern nu={nu}')
    except:
        pass
    try:
        plt.plot(x,np.squeeze(grad(x,np.array([[0.]]),1.)),label='gradient')
    except:
        pass
    plt.legend()
    plt.savefig(f'./pic/matern_nu_{string}.png')
    plt.close()

def check_matern_sklearn(nu):
    x = jnp.atleast_2d(jnp.linspace(-10.,10.,100)).T
    skmatern = sk_Matern(length_scale=1.,nu=nu)
    return skmatern(x,np.array([[0],[1]]))

def check_matern_kernelmtd(nu):
    x = jnp.atleast_2d(jnp.linspace(-10.,10.,100)).T
    matk = MaternKernel(l=1.,nu=nu)
    return matk.kde(x.reshape(100,1,1),np.array([[0],[1]]).reshape(2,1,1)).reshape(100,2)

def test_equivalent_function():
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

def main():
    test_matern_nu(0.5)
    test_matern_nu(1.5)
    test_matern_nu(2.5)
    test_matern_nu(np.inf)
    test_equivalent_function()

if __name__ == '__main__':
    main()