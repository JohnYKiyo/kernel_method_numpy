from ..metrics.pairwise import euclidean_distances,mahalanobis_distances
from .gauss_kernel import gauss_kernel

from sklearn.model_selection import KFold
import numpy as np
import scipy as scp
import pandas as pd
import optuna

class band_width():
    '''
    Determine bandwidth of gaussian kernel.

    method: str
        - Median heuristic
        - Scott heuristic [1]
        - Silverman heuristic [2]
        - LSCV [3-5]
        - LCV

    data: array_like
        Datapoints to estimate.
        1D-array: [x_1,x_2,...,x_n]
        1D-array input is treated as 1-dimensional data
        2D-array: [[x_1^d1,x_1^d2], [x_2^d1,x_2^d2],...,[x_n^d1,x_n^d2]]

    weights: array_like
        weights of datapoints. This must be the same shape as dataset.

    Notes:
        Bandwidth selection strongly affects kernel density estimation.
    [1] D.W. Scott, “Multivariate Density Estimation: Theory, Practice, and Visualization”, John Wiley & Sons, New York, Chicester, 1992.
    [2] B.W. Silverman, “Density Estimation for Statistics and Data Analysis”, Vol. 26, Monographs on Statistics and Applied Probability, Chapman and Hall, London, 1986.
    [3] P. Hall, “Large Sample Optimality of Least Squares Cross-Validation in Density Estimation,” Ann. Stat., vol. 11, no. 4, pp. 1156–1174, 1983. 
    [4] C. J. Stone, “An Asymptotically Optimal Window Selection Rule for Kernel Density Estimates,” Ann. Stat., vol. 12, no. 4, pp. 1285–1297, 1984.
    [5] W. Härdle, P. Hall, and J. S. Marron, “How far are automatically chosen regression smoothing parameters from their optimum?,” J. Am. Stat. Assoc., vol. 83, no. 401, pp. 86–95, 1988.
    '''

    def __init__(self, data, method=None, weights=None):
        if isinstance(data,list):
            data = np.array(data)
        if isinstance(data,pd.DataFrame):
            data = data.values
        if data.ndim ==1: #1D-array
            data = data.reshape(-1,1) # 1D-array [x1,...,xn] as data having 1D feature. -> [[x1], [x2],...,[xn]]
        self._data = data
        self._Ndata, self._ndim = self._data.shape
        if not self._Ndata > 1:
            raise ValueError("'dataset' input should have multiple elements.")

        if weights is not None:
            self._weights = np.atleast_1d(weights).astype(float)
            self._weights /= np.sum(self._weights)
            if self._weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(self._weights) != self._Ndata:
                raise ValueError("'weights' input should be of length the number of datapoints.")
        else:
            self._weights = np.atleast_1d(np.ones(self._Ndata)/self._Ndata) #1/N

        self._neff = 1./np.sum(self._weights**2) # if weight is None -> neff = N
        self._compute_covariance()
        self.set_bandwidth(method=method)

    def _compute_covariance(self):
        epsilon=1e-12
        lambdaI = epsilon*np.eye(self._ndim)
        if not hasattr(self, '_data_inv_cov'):
            self._data_cov = np.atleast_2d(np.cov(self._data,rowvar=False,
                                                  bias=False,
                                                  aweights=self._weights))
            try:
                self._data_inv_cov = np.linalg.inv(self._data_cov)
            except:
                self._data_cov += lambdaI
                self._data_inv_cov = np.linalg.inv(self._data_cov+lambdaI)

    def set_bandwidth(self, method=None):
        if (method is None) or (method == 'cov'):
            self._method = 'cov'
            self._covariance_factor = lambda: 1.0 # band width is std of data.
        elif method == 'median':
            self._method = method
            self._covariance_factor = self.median_method
        elif method == 'scott':
            self._method = method
            self._covariance_factor = self.scott_factor
        elif method == 'silverman':
            self._method = method
            self._covariance_factor = self.silverman_factor
        elif method == 'LSCV':
            """
            minimizing Integrated Mean Square Error (IMSE) method.
            """
            self._method = method
            raise KeyError('Sorry, Least Square CV method have not implemented yet.')
        elif method == 'LCV':
            """
            minimizing Likelifood method.
            """
            self._method = method
            self._covariance_factor = self.LCV_method
        elif np.isscalar(method) and not isinstance(method, str):
            self._method = 'use constant'
            self._covariance_factor = lambda: method
        else:
            msg = "'method' should be 'cov', 'scott', 'silverman', 'scalar', 'median', 'LCV','LSCV'"
            raise ValueError(msg)

        self._compute_corrected_cov()

    def _compute_corrected_cov(self):
        self._factor = self._covariance_factor()
        if self._method not in ['median', 'LCV']:
            self._cov = self._data_cov * self._factor**2
            self._inv_cov = self._data_inv_cov / self._factor**2
        self._normalize = np.sqrt(np.linalg.det(2*np.pi*self.cov))

    def scott_factor(self):
        """
        Compute Scott's factor.
        : n ^ (-1/(d+4))
        """
        return np.power(self._neff, -1./(self._ndim+4))

    def silverman_factor(self):
        """
        Compute the Silverman's factor.
        factor = (4/(d+2))**(1/(d+4)) * n**(-1/(d+4))
        so, j-th bandwidth
        h_j = factor*sig_j
        """
        return np.power(self._neff*(self._ndim+2.0)/4.0, -1./(self._ndim+4))

    def median_method(self):
        '''
        The Euclidean distance between data is calculated, and the mode of distance is defined as the bandwidth.
        '''
        dists = euclidean_distances(self._data, squared=True)
        h = np.median(np.sqrt(dists))
        if self._method == 'median':
            """Prevents the calculated cov from changing when this function is executed."""
            self._cov = np.eye(self._ndim)*h
            self._inv_cov = np.eye(self._ndim)/h
        return h

    def LCV_method(self):
        if self._method == 'LCV':
            lcv = LCV(self._data,self._ndim)
            lcv.compute(n_trials=150,#trial shuld increase as dimentions increase
                        pruner='SHM',
                        vervose=True,
                        min_resource=5,
                        reduction_factor=2,
                        min_early_stopping_rate=0)
            self._cov = lcv.cov
            self._inv_cov = np.linalg.inv(self._cov)
            del lcv

    def __call__(self):
        return self.bandwidth(save_dim=False,squared=False)

    def bandwidth(self, save_dim=False,squared=True):
        '''
        general band-width is calculated by the law of propagation of error.
        '''
        if save_dim:
            val = np.diag(self.cov)
        else:
            val = np.diag(self.cov).sum() #\sigma = \sqrt(\sigma_1^2 + \sigma_2^2 + ...+\sigma_d^2))

        if squared:
            val = np.sqrt(val)

        return val

    @property
    def cov(self):
        return self._cov

    @property
    def inv_cov(self):
        return self._inv_cov


class LCV():
    def __init__(self,Data,n_params):
        self._data = Data
        self._nparams = n_params

    def LOOLL(self,test,train,kernel):
        epsilon=1e-12
        return np.log(np.average(kernel.pdf(test,train,normalize=True),axis=1)+epsilon)

    def __call__(self, n_trials, pruner=None, vervose=False, **args):
        self.compute(n_trials,pruner,vervose,**args)
        
    def compute(self,n_trials,pruner=None,vervose=True,**args):
        if pruner=='SHM':
            pruner_method = optuna.pruners.SuccessiveHalvingPruner(**args)
        if not vervose:
            optuna.logging.disable_default_handler()
        study = optuna.create_study(pruner=pruner_method)
        study.optimize(self._objective(),n_trials=n_trials)
        self._study = study
        self._cov = np.diag(np.fromiter(study.best_params.values(),dtype=float))

    #this is for TPE+Successive Halving Algorithm.
    def _objective(self):
        kf = KFold(n_splits=len(self._data), shuffle=True)
        min_search= 1e-12
        max_search= 10**(np.log10(self._data.max(axis=0)-self._data.min(axis=0)+10).astype(int)+1)

        def obj(trial):
            epsilon=1e-12
            cov = []
            for i in range(self._nparams):
                cov.append(trial.suggest_loguniform(f'h{i}', min_search, max_search[i]))
            cov = np.diag(cov)

            k = gauss_kernel(covariance=cov,n_features=self._nparams)
            losses = []
            for i, (train_idx,test_idx) in enumerate(kf.split(self._data)):
                losses.append(self.LOOLL(self._data[test_idx],self._data[train_idx],k))
                if i%100==0:
                    trial.report(-1*np.mean(losses),i)
                    if trial.should_prune(i):
                        raise optuna.exceptions.TrialPruned()
            return -1*np.mean(losses)
        return obj

    @property
    def cov(self):
        if not hasattr(self,'_cov'):
            self.__call__(n_trials=50)
        return self._cov