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

if __name__ == '__main__':
    import sys
    sys.path.append('./')
    from kernelmtd.kernel.matern import *
    import os
    if not os.path.exists('./pic/'):
        os.makedirs('./pic/')

    import numpy as np
    import matplotlib.pyplot as plt

    main()
