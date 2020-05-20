from .gauss_kernel import gauss_kernel

class kernel_gram_matrix(gauss_kernel):
    def __init__(self,x,cov):
        super().__init__(covariance=cov, n_features=x.shape[1])
        self._gram_matrix = self.pdf(x,x,normalize=False)
        self._log_gram_matrix = self.logpdf(x,x,normalize=False)

    def __call__(self):
        return self.gram_matrix

    @property
    def gram_matrix(self):
        return self._gram_matrix
    @property
    def log_gram_matrix(self):
        return self._log_gram_matrix
