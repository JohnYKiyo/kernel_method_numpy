from ...KernelMean import KernelMean
from ..functions import gauss_kernel
import numpy as np

def MaximumMeanDiscrepancy(X,Y, kernel=None):
    if kernel is not None:
        '''
        Compute Maximum Mean Discrepancy:
        ||m_X-m_Y||^2 = 1/(N^2) k(X,X) + 1/(M^2) k(Y,Y) - 2*1/(NM) k(X,Y)
        '''
        MMD = kernel.pdf(X,X).mean()+kernel.pdf(Y,Y).mean()-2*kernel.pdf(X,Y).mean()
        return MMD

    if not isinstance(X,KernelMean):
        raise TypeError(f"X shuld be KernelMean class, but got {type(X)}")
    if not isinstance(Y,KernelMean):
        raise TypeError(f"Y shuld be KernelMean class, but got {type(Y)}")

    X_samples = X.data.values
    Y_samples = Y.data.values
    kernel_X = X.kernel
    kernel_Y = Y.kernel
    weights_X = X.weights.copy()
    weights_Y = Y.weights.copy()
    if (kernel_X.cov != kernel_Y.cov).any():
        raise ValueError(f"X kernel and Y kernel must be the same")

    MMD = np.dot(weights_X, np.dot(weights_X,kernel_X.pdf(X_samples,X_samples)))\
        + np.dot(weights_Y, np.dot(weights_Y,kernel_X.pdf(Y_samples,Y_samples)))\
        - 2*np.dot(weights_Y,np.dot(weights_X,kernel_X.pdf(X_samples,Y_samples)))
    return MMD

def MaximumMeanDiscrepancy_of_normal_pdf_and_kernel_mean(kernelmean, mu, sigma):
    if not isinstance(kernelmean, KernelMean):
        raise TypeError(f"kernelmean shuld be KernelMean class, but got {type(kernelmean)}")

    w = kernelmean.weights
    cov_kernel_mean = kernelmean.kernel.cov
    samples = kernelmean.data.values
    if cov_kernel_mean.shape != (1,1):
        raise TypeError(f"kernelmean covariance shuld be scolar.")

    K = gauss_kernel.gauss_kernel(n_features=1, covariance=(sigma**2+cov_kernel_mean))
    val = K.pdf(samples,np.atleast_2d(mu))
    cross_term = -2*np.dot(w,val)*np.sqrt(cov_kernel_mean)/np.sqrt(cov_kernel_mean+sigma**2)
    normal_pdf_term = np.sqrt(cov_kernel_mean)/np.sqrt(2*sigma**2+cov_kernel_mean)
    
    return np.dot(w, np.dot(w, kernelmean.kernel.pdf(samples,samples))) + cross_term + normal_pdf_term