def test_nu(nu=0.5):
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

def main():
    test_nu(0.5)
    test_nu(1.5)
    test_nu(2.5)
    test_nu(np.inf)
    #test_equivalent_function()

if __name__ == '__main__':
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

    main()

    #cannot use a timeit function in module.    
    #nu=0.5
    skmatern = sk_Matern(length_scale=1.,nu=0.5)
    matk = MaternKernel(l=1.,nu=0.5)
    x = jnp.atleast_2d(jnp.linspace(-10.,10.,100)).T
    np.testing.assert_almost_equal(
        skmatern(x,np.array([[0],[1]])),
        matk.kde(x.reshape(100,1,1),np.array([[0],[1]]).reshape(2,1,1)).reshape(100,2))

    print(f'nu=0.5 sklearn:{timeit.timeit("skmatern(x,jnp.array([[0],[1]]))", globals = globals(), number=1000)*1000:.2f} us')
    print(f'nu=0.5 kernelmtd:{timeit.timeit("matk.kde(x.reshape(100,1,1),jnp.array([[0],[1]]).reshape(2,1,1))", globals = globals(), number=1000)*1000:.2f} us')
    
    #nu=1.5
    skmatern = sk_Matern(length_scale=1.,nu=1.5)
    matk = MaternKernel(l=1.,nu=1.5)
    x = jnp.atleast_2d(jnp.linspace(-10.,10.,100)).T
    np.testing.assert_almost_equal(
        skmatern(x,np.array([[0],[1]])),
        matk.kde(x.reshape(100,1,1),np.array([[0],[1]]).reshape(2,1,1)).reshape(100,2))
    print(f'nu=1.5 sklearn:{timeit.timeit("skmatern(x,jnp.array([[0],[1]]))", globals = globals(), number=1000)*1000:.2f} us')
    print(f'nu=1.5 kernelmtd:{timeit.timeit("matk.kde(x.reshape(100,1,1),jnp.array([[0],[1]]).reshape(2,1,1))", globals = globals(), number=1000)*1000:.2f} us')
    
    #test nu=2.5
    skmatern = sk_Matern(length_scale=1.,nu=2.5)
    matk = MaternKernel(l=1.,nu=2.5)
    x = jnp.atleast_2d(jnp.linspace(-10.,10.,100)).T
    np.testing.assert_almost_equal(
        skmatern(x,np.array([[0],[1]])),
        matk.kde(x.reshape(100,1,1),np.array([[0],[1]]).reshape(2,1,1)).reshape(100,2))
    print(f'nu=2.5 sklearn:{timeit.timeit("skmatern(x,jnp.array([[0],[1]]))", globals = globals(), number=1000)*1000:.2f} us')
    print(f'nu=2.5 kernelmtd:{timeit.timeit("matk.kde(x.reshape(100,1,1),jnp.array([[0],[1]]).reshape(2,1,1))", globals = globals(), number=1000)*1000:.2f} us')

    #test nu=inf
    skmatern = sk_Matern(length_scale=1.,nu=np.inf)
    matk = MaternKernel(l=1.,nu=np.inf)
    x = jnp.atleast_2d(jnp.linspace(-10.,10.,100)).T
    np.testing.assert_almost_equal(
        skmatern(x,np.array([[0],[1]])),
        matk.kde(x.reshape(100,1,1),np.array([[0],[1]]).reshape(2,1,1)).reshape(100,2))
    print(f'nu=inf sklearn:{timeit.timeit("skmatern(x,jnp.array([[0],[1]]))", globals = globals(), number=1000)*1000:.2f} us')
    print(f'nu=inf kernelmtd:{timeit.timeit("matk.kde(x.reshape(100,1,1),jnp.array([[0],[1]]).reshape(2,1,1))", globals = globals(), number=1000)*1000:.2f} us')
    
    #test nu=ohter(=5)
    skmatern = sk_Matern(length_scale=1.,nu=5)
    matk = MaternKernel(l=1.,nu=5)
    x = jnp.atleast_2d(jnp.linspace(-10.,10.,100)).T
    np.testing.assert_almost_equal(
        skmatern(x,np.array([[0],[1]])),
        matk.kde(x.reshape(100,1,1),np.array([[0],[1]]).reshape(2,1,1)).reshape(100,2))
    print(f'nu=other (=5.) sklearn:{timeit.timeit("skmatern(x,jnp.array([[0],[1]]))", globals = globals(), number=1000)*1000:.2f} us')
    print(f'nu=other (=5.) kernelmtd:{timeit.timeit("matk.kde(x.reshape(100,1,1),jnp.array([[0],[1]]).reshape(2,1,1))", globals = globals(), number=1000):.2f} ms')