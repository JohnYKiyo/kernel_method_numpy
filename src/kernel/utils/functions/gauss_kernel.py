from ..metrics.pairwise import mahalanobis_distances
import numpy as np
import scipy as scp

class gauss_kernel():
    def __init__(self,covariance, n_features):
        if np.isscalar(covariance):
            if covariance == 0:
                covariance = 10e-8
            covariance = np.diag(np.full(n_features,covariance))
        if isinstance(covariance,list):
            covariance = np.array(covariance)
            if covariance.ndim == 1:
                covariance = np.diag(covariance)
        if covariance.shape != (n_features,n_features):
            raise ValueError(f"Covariance matrix should be symmetric matrix.(n_features,n_features), but given {covariance.shape}")

        self._n_features = n_features
        self._cov = covariance
        self._inv_cov = np.linalg.inv(covariance)
        self._calculate_normalize_factor()

    def __call__(self,x,y,normalize=False):
        return self.pdf(x,y,normalize=normalize)

    def logpdf(self,x,y,normalize=False):
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        if (x.shape[1] != self._n_features) or (y.shape[1] != self._n_features):
            raise ValueError(f"The features dimention (x,y):({x.shape[1]},{y.shape[1]})" \
                                f"should be same as ({self._n_features},{self._n_features}).")
        val = -0.5*mahalanobis_distances(x,y,Q=self._inv_cov,squared=True)
        if normalize:
            val += -1.*np.log(self._norm_factor)
        return val

    def pdf(self,x,y,normalize=False):
        """
                            np.exp(-0.5(x-y)^T Q (x-y))
            val  =  ------------------------------------------------
                        ((2*np.pi*self.sigma**2)**(dim/2.))
        """
        return np.exp(self.logpdf(x,y,normalize))

    def _calculate_normalize_factor(self):
        """
        nom_factor = np.sqrt((2*pi)^k |S|)
        """
        self._norm_factor = np.sqrt(np.linalg.det(2*np.pi*self._cov))

    @property
    def norm_factor(self):
        return self._norm_factor
    @property
    def cov(self):
        return self._cov
    @property
    def inv_cov(self):
        return self._inv_cov

    def grad_logpdf(self, x,y):
        """
        \grad log_pdf = 2 Q (x-y)
        numpy technique memo:
            pairwise subtraction x[:,None,:] - y[None,:,:] 
            [[[x1-y1],[x1-y2],...]
            [[x2-y1],[x2-y2],...]]
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        return -1*np.dot(self._inv_cov,
                            (x[:,np.newaxis,:]-y[np.newaxis,:,:]).transpose(0,2,1)).transpose(1,2,0)

    def grad(self, x, y,normalize=False):
        x = np.atleast_2d(x) #convert [x,x,...,x] to [[x,x,...,x]
        y = np.atleast_2d(y) #convert [y,y,...,y] to [[y,y,...,y]]

        return np.einsum('ij,ijk->ijk',self.pdf(x,y,normalize),self.grad_logpdf(x,y))
