from ..utils import transform_data
from ..metrics import pairwise_euclid_distances
from ..kernel import GaussKernel

import numpy as np


class Bandwidth(object):
    """bandwidth selection for rbf-kernel hyperparameter.

    Args:
        cov (ndarray): [description]
        inv_cov (ndarray): [description]
        bandwidth (ndarray): [description]

    Raises:
        ValueError: [description]
        ValueError: [description]
        NotImplementedError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]

    Notes:
        Bandwidth selection strongly affects kernel density estimation.
        [1] D.W. Scott, “Multivariate Density Estimation: Theory, Practice, and Visualization”, John Wiley & Sons, New York, Chicester, 1992.
        [2] B.W. Silverman, “Density Estimation for Statistics and Data Analysis”, Vol. 26, Monographs on Statistics and Applied Probability, Chapman and Hall, London, 1986.
        [3] P. Hall, “Large Sample Optimality of Least Squares Cross-Validation in Density Estimation,” Ann. Stat., vol. 11, no. 4, pp. 1156–1174, 1983.
        [4] C. J. Stone, “An Asymptotically Optimal Window Selection Rule for Kernel Density Estimates,” Ann. Stat., vol. 12, no. 4, pp. 1285–1297, 1984.
        [5] W. Härdle, P. Hall, and J. S. Marron, “How far are automatically chosen regression smoothing parameters from their optimum?,” J. Am. Stat. Assoc., vol. 83, no. 401, pp. 86–95, 1988.
    """

    def __init__(self, data, method='scott', weights=None):
        """[summary]

        Args:
            data (array-like):
                Datapoints to estimate.
                1D-array [x_1,x_2,...,x_n]. This input is treated as 1-dimensional data.
                2D-array [[x_1^d1,x_1^d2], [x_2^d1,x_2^d2],...,[x_n^d1,x_n^d2]].

            method (str or float, optional):
                Defaults to 'scott'.
                Select the bandwidth selection method from　['cov', 'scott', 'silverman', 'median', 'LCV','LSCV'] .
                - Median heuristic
                - Scott heuristic [1]
                - Silverman heuristic [2]
                - LSCV [3-5]
                - LCV [3-5]
                - covariance of data

            weights (array-like, optional):
                Defaults to None. Weights of datapoints. This must be the same shape as dataset.

        Raises:
            ValueError: [description]
        """
        self.__data = transform_data(data)
        self.__n_data, self.__ndim = self.__data.shape

        if weights is not None:
            self.__weights = np.atleast_1d(weights).astype(float)
            self.__weights /= np.sum(self.__weights)
            if (self.__weights.ndim != 1) or (self.__weights.size != self.__n_data):
                raise ValueError("`weights` input should be one-dimensional or the length of the number of datapoints.")
        else:
            self.__weights = np.ones(self.__n_data) / self.__n_data  # 1/N

        self.__neff = 1. / np.sum(self.__weights**2)  # if weight is None -> neff = N
        self.__compute_covariance()
        self.set_bandwidth(method=method)

    def __compute_covariance(self):
        epsilon = 1e-12
        lambdaI = epsilon * np.eye(self.__ndim)
        if not hasattr(self, '__data_inv_cov'):
            self.__data_cov = np.atleast_2d(np.cov(self.__data, rowvar=False, bias=False, aweights=self.__weights))

            try:
                self.__data_inv_cov = np.linalg.inv(self.__data_cov)

            except:  # noqa: E722
                self.__data_cov += lambdaI
                self.__data_inv_cov = np.linalg.inv(self.__data_cov)

    def set_bandwidth(self, method=None):
        if (method is None) or (method == 'cov'):
            self.__method = 'cov'
            self.__covariance_factor = lambda: 1.0  # band width is std of data.

        elif method == 'median':
            self.__method = method
            self.__covariance_factor = self.median_method

        elif method == 'scott':
            self.__method = method
            self.__covariance_factor = self.scott_factor

        elif method == 'silverman':
            self.__method = method
            self.__covariance_factor = self.silverman_factor

        elif method == 'LSCV':
            """
            minimizing Least Square Cross Validation.
            """
            raise NotImplementedError

        elif method == 'LCV':
            """
            maximizing Likelihood Cross Validation.
            """
            self.__method = method
            self.__covariance_factor = self.LCV_method

        elif np.isscalar(method) and not isinstance(method, str):
            self.__method = 'constant'
            self.__covariance_factor = lambda: method

        else:
            msg = "'method' should be 'cov', 'scott', 'silverman', 'scalar', 'median', 'LSCV'"
            raise ValueError(msg)

        self.__compute_cov()

    def __compute_cov(self):
        self.__factor = self.__covariance_factor()
        if self.__method not in ['median', 'LSCV', 'LCV']:
            self._cov = self.__data_cov * self.__factor**2
            self._inv_cov = self.__data_inv_cov / self.__factor**2

    def scott_factor(self):
        """
        Compute Scott's factor.
        : n ^ (-1/(d+4))
        """
        return np.power(self.__neff, -1. / (self.__ndim + 4))

    def silverman_factor(self):
        """
        Compute the Silverman's factor.
        factor = (4/(d+2))**(1/(d+4)) * n**(-1/(d+4))
        so, j-th bandwidth
        h_j = factor*sig_j
        """
        return np.power(self.__neff * (self.__ndim + 2.0) / 4.0, -1. / (self.__ndim + 4))

    def median_method(self):
        '''
        The Euclidean distance between data is calculated, and the mode of distance is defined as the bandwidth.
        '''
        dists = pairwise_euclid_distances(self.__data, self.__data, True)
        ind = np.triu_indices(self.__n_data, k=1)
        h = np.median(dists[ind])
        if self.__method == 'median':
            """Prevents the calculated cov from changing when this function is executed."""
            self._cov = np.eye(self.__ndim) * h
            self._inv_cov = np.eye(self.__ndim) / h
        return h

    def LCV_method(self):
        if self.__method == 'LCV':
            # try:
            #     import optuna
            # except ImportError:
            #     print("Cannot find optuna library, please install it to use this option.")
            #     raise
            #
            # optuna.logging.disable_default_handler()
            # def LogLikelihood(trial):
            #     epsilon=1e-12
            #     cov = []
            #     for i in range(2):
            #         cov.append(trial.suggest_loguniform(f'h{i}', 1e-3, 1e+2))
            #     cov = np.diag(cov)
            #     M = gauss_kernel(self.__data,self.__data,cov=cov)
            #     return np.log((M-np.diag(np.diag(M))).mean(axis=1)+epsilon).mean().ravel()
            #
            # study = optuna.create_study(direction='maximize')
            # study.optimize(LogLikelihood,n_trials=100)
            # self._cov = np.diag(np.fromiter(study.best_params.values(),dtype=float))
            # self._inv_cov = np.linalg.inv(self._cov)
            try:
                import gpbayesopt
            except ImportError:
                print("Cannot find gpbayesopt library, please install it to use this option. \n \
                      Install: pip install git+https://github.com/JohnYKiyo/bayesian_optimization.git")
                raise

            kernel = GaussKernel(covariance=np.diag(np.ones(self.__ndim)))

            def LogLikelihood(cov):
                # print(cov,kernel.cov)
                epsilon = 1e-12
                kernel.cov = np.diag(cov.ravel())
                M = kernel(self.__data, self.__data, normalize=True)
                return np.log((M - np.diag(np.diag(M))).mean(axis=1) + epsilon).mean().ravel()

            bo = gpbayesopt.BayesOpt(LogLikelihood,
                                     initial_input=np.atleast_2d(np.ones(self.__ndim)),
                                     alpha=1e-6,
                                     kernel=gpbayesopt.kernel.MaternKernel(),
                                     acq=gpbayesopt.acquisition.UCB,
                                     acq_optim=gpbayesopt.acquisition_optimizer.Acquisition_L_BFGS_B_LogOptimizer(bounds=np.array([[-3, 2] for i in range(self.__ndim)]), n_trial=2),
                                     maximize=True,
                                     function_input_unpacking=False)
            bo.run_optim(50)
            self._cov = np.diag(bo.best_params)
            self._inv_cov = np.linalg.inv(self._cov)
            return 1.

    @property
    def cov(self):
        return self._cov

    @property
    def inv_cov(self):
        return self._inv_cov

    @property
    def bandwidth(self):
        return np.sqrt(np.diag(self._cov).sum())

    def __repr__(self):
        val = f'Bandwidth selection method: {self.__method}\n'
        val += f'Number of data: {self.__n_data}\n'
        val += f'Dimensions: {self.__ndim}\n'
        val += f'Covariance matrix:\n {self.cov}\n'
        val += f'Bandwidth: {self.bandwidth}\n'
        return val
