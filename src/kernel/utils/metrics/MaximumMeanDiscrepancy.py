from ...KernelMean import KernelMean
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
