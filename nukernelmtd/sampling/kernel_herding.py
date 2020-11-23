from functools import partial
import numpy as np
from scipy.optimize import minimize
from scipy import random
from tqdm import tqdm
import warnings

from ..kernel_mean import KernelMean
from ..kernel import GaussKernel


class KernelHerding(object):
    """[summary]
    The class of Kernel Herding. Sampling from Kernel Mean.
    Args:
        KernelMeanObj (KernelMean): KernelMean.
    Note:
        [1] Chen, Y., Welling, M., & Smola, A. (2010). Super-samples from kernel herding. Proceedings of the 26th Conference on Uncertainty in Artificial Intelligence, UAI 2010, 109–116.
    """
    def __init__(self, KernelMeanObj):
        if not isinstance(KernelMeanObj, KernelMean):
            raise TypeError(f"init object shuld be KernelMean class, but got {type(KernelMeanObj)}")

        self.__KernelMean = KernelMeanObj
        self.__kernel = KernelMeanObj.kernel
        self.__samples = None

    def sampling(self, sample_size, **kwargs):
        """[summary]
        Args:
            sample_size (int): sample size of herding.
            **kwargs:
                normalize (bool, optional): Defaults to False.
                    Specify `True` when normalizing the kernel. Some kernels do not have a normalization option, so see the kernel's docstrings.

                weights_normalize (bool, optional): Defaults to False.
                    Specify `True` when normalizing kernel average weighting.

                max_trial (int, optional): Defaults to 2.
                    Search by changing the initial value by the number of max_trial.

                derivatives (bool, optional): Defaults to False.
                    If the derivative is defined in the kernel, the search is performed using the derivative value.

                The remainder of the keyword arguments are inherited from
                `scipy.optimize.minimize`, and their descriptions are copied here for convenience.

                method (str, optional): Defaults to 'L-BFGS-B'.
                    Type of solver.  Should be one of
                    - 'Nelder-Mead' :ref:`(see here) <optimize.minimize-neldermead>`
                    - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
                    - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
                    - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
                    - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
                    - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
                    - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
                    - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
                    - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
                    - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
                    - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
                    - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
                    - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
                    - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
                    - custom - a callable object (added in version 0.14.0),
                see below for description.
                If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,depending if the problem has constraints or bounds.

        Example:
            >>> from kernelmtd.data import testdata
            >>> bw = Bandwidth(data=testdata,method='scott')
            >>> KM = KernelMean(data=testdata,kernel=GaussKernel(sigma=bw.bandwidth))
            >>> KH = KH = KernelHerding(KM)
            >>> supersamples = KH.sampling(sample_size=10,max_trial=2)
            >>> KH.sampling(sample_size=10,max_trial=2,derivatives=True)

        Returns:
            ndarray : supersamples.
        """
        normalize = kwargs.pop('normalize', False)
        weights_normalize = kwargs.pop('weights_normalize', False)
        max_trial = kwargs.pop('max_trial', 2)
        derivatives = kwargs.pop('derivatives', False)

        self.__samples = None
        h = partial(self.__KernelMean.kde, normalize=normalize, weights_normalize=weights_normalize)
        grad_h = None
        if derivatives:
            try:
                self.__KernelMean.gradkde(self.__KernelMean.data.iloc[0:2, :].values)  # check gradkde is defined.
                derivatives = True
                grad_h = partial(self.__KernelMean.gradkde, normalize=normalize, weights_normalize=weights_normalize)
            except:  # noqa
                derivatives = False
                warnings.warn('kernel has no gradkde method. So, run with derivative option False', Warning)
        with tqdm(total=sample_size) as bar:
            x1, _ = self.__argmax(h, grad_h, derivatives=derivatives, max_trial=max_trial, **kwargs)
            samples = np.atleast_2d(x1)
            bar.update()
            for idx in range(sample_size - 1):
                h, h_prime = self.__herding_update(samples, normalize=normalize, weights_normalize=weights_normalize, derivatives=derivatives, **kwargs)
                x, _ = self.__argmax(h, h_prime, derivatives=derivatives, max_trial=max_trial, **kwargs)
                samples = np.append(samples, np.atleast_2d(x), axis=0)
                self.__samples = samples

                post_str = f'param:{x}'
                bar.set_description_str('KernelHerding')
                bar.set_postfix_str(post_str)
                bar.update()
        return self.__samples

    def __herding_update(self, samples, normalize=False, weights_normalize=False, derivatives=False, **kwargs):
        def fnc(x):
            kmean = self.__KernelMean.kde(x, normalize=normalize, weights_normalize=weights_normalize)
            phi = np.mean(self.__kernel.kde(x, samples, normalize=normalize), axis=1, keepdims=True)
            return kmean - phi
        f_prime = None
        if derivatives:
            def f_prime(x):
                m_prime = self.__KernelMean.gradkde(x, weights_normalize=weights_normalize, normalize=normalize)
                phi_prime = np.mean(self.__kernel.gradkde(x, samples, normalize=normalize), axis=1, keepdims=True)
                return m_prime - phi_prime
        return fnc, f_prime

    def __argmax(self, h, h_prime, derivatives=False, max_trial=2, **kwargs):
        minus_h = lambda x: np.array(np.squeeze(-1. * h(np.atleast_2d(x))))  # noqa : E731 do not assign a lambda expression, use a def.
        minus_h_prime = None
        if derivatives:
            minus_h_prime = lambda x: np.array(np.squeeze(-1. * h_prime(np.atleast_2d(x))))  # noqa : E731 do not assign a lambda expression, use a def.
        x, val = self.__optimizer_scipy(minus_h, minus_h_prime, derivatives=derivatives, max_trial=max_trial, **kwargs)
        return x, val

    def __optimizer_scipy(self, h, h_prime, derivatives=False, max_trial=2, **kwargs):
        kwargs['method'] = kwargs.pop('method', 'L-BFGS-B')
        if 'x0' not in kwargs:
            kwargs['x0'] = random.uniform(low=np.min(self.__KernelMean.data, axis=0), high=np.max(self.__KernelMean.data, axis=0))
        if derivatives:
            kwargs['jac'] = h_prime

        x, val = kwargs['x0'], np.inf
        for i in range(max_trial):
            optimize_success = False
            while not optimize_success:
                optimize_result = minimize(h, **kwargs)
                if not optimize_result.success:
                    kwargs['x0'] = random.uniform(low=np.min(self.__KernelMean.data, axis=0), high=np.max(self.__KernelMean.data, axis=0))
                    continue
                else:
                    optimize_success = True

            if optimize_result.fun < val:
                x = optimize_result.x
                val = optimize_result.fun
        return x, val

    @property
    def supersamples(self):
        if self.__samples is None:
            raise ValueError('super samples do not exist.')
        return self.__samples


def test(data):
    import numpy as np
    import matplotlib.pyplot as plt

    KM = KernelMean(data=data.testdata, kernel=GaussKernel(sigma=1.))
    KH = KernelHerding(KM)
    supersamples = KH.sampling(sample_size=50, max_trial=2, derivatives=False)
    KM_herding = KernelMean(data=supersamples, kernel=GaussKernel(sigma=1.))

    x = np.arange(-10.0, 10.0, 0.1)
    y = np.arange(-10.0, 10.0, 0.1)
    xx, yy = np.meshgrid(x, y)
    z = KM.kde(np.array([xx.ravel(), yy.ravel()]).T)
    zz = z.reshape(xx.shape[0], -1)
    z_h = KM_herding.kde(np.array([xx.ravel(), yy.ravel()]).T)
    zz_h = z_h.reshape(xx.shape[0], -1)

    fig = plt.figure(figsize=[10, 5])
    ax = fig.add_subplot(121)
    bar = ax.contour(xx, yy, zz, 20, alpha=0.5, cmap='jet')
    ax.scatter(KM.data.iloc[:, 0], KM.data.iloc[:, 1], alpha=1, marker='.', color='C1', label='data')
    ax.set_title('original samples kernel mean')
    ax.axis("image")
    cax = fig.colorbar(bar, ax=ax)
    ax_pos = ax.get_position()
    cax_pos = cax.ax.get_position()
    cax_pos = [cax_pos.x0, ax_pos.y0, cax_pos.x1 - cax_pos.x0, ax_pos.y1 - ax_pos.y0]
    cax.ax.set_position(cax_pos)
    ax.legend()

    ax = fig.add_subplot(122)
    bar = ax.contour(xx, yy, zz_h, 20, alpha=0.5, cmap='jet')
    ax.scatter(KM_herding.data.iloc[:, 0], KM_herding.data.iloc[:, 1], alpha=1, marker='.', color='C1', label='data')
    ax.set_title('supersamples kernel mean')
    ax.axis("image")
    cax = fig.colorbar(bar, ax=ax)
    ax_pos = ax.get_position()
    cax_pos = cax.ax.get_position()
    cax_pos = [cax_pos.x0, ax_pos.y0, cax_pos.x1 - cax_pos.x0, ax_pos.y1 - ax_pos.y0]
    cax.ax.set_position(cax_pos)
    ax.legend()
    fig.savefig('./pic/test/kernelherding/supersample.png')
    fig.clear()

    mmd_herding = []
    mmd_random = []
    for i in range(1, 50):
        mmd_herding.append(metrics.maximum_mean_discrepancy(X=KM.data.values,
                                                            Y=KM_herding.data.values[:i],
                                                            kernel=KM.kernel))
        mmd_random.append(metrics.maximum_mean_discrepancy(X=KM.data.values,
                                                           Y=KM.data.values[:i],
                                                           kernel=KM.kernel))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(mmd_herding, label='herding')
    ax.plot(mmd_random, label='random')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('sample')
    ax.set_ylabel(r'$||\mu - \mu_p||^2$')
    plt.legend()
    fig.savefig('./pic/test/kernelherding/mmd.png')


if __name__ == '__main__':
    from kernelmtd import data, metrics
    import os
    if not os.path.exists('./pic/test/kernelherding/'):
        os.makedirs('./pic/test/kernelherding/')
    test(data)
