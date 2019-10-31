from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import scipy as scp

def get_band_width(d, method='median'):
    '''
    Determine bandwidth of gaussian kernel. 
    Method:
        - Median heuristic
        - Silverman heuristic
        - Scott heuristic
        - LSCV
    '''
    
    if method == 'median':
        K = euclidean_distances(d, squared=True) #データ間のユークリッド距離を総当たり的に求めてる
        sigma = np.median(np.sqrt(K)) #最頻する距離で正規化するためにmedian取ってる
        
    elif method in ['silverman', 'scott']:
        factor = 1.06 if method == 'silverman' else 0.9
        distances_from_mean = euclidean_distances(d,[d.mean(axis=0)], squared=False)
        std = np.sqrt(np.square(distances_from_mean).mean())
        sig = min(std,\
                  (np.percentile(distances_from_mean,75)-np.percentile(distances_from_mean,25))/1.34)
        sigma = factor*sig/(d.shape[0]**(1./5))
        
    elif method == 'LSCV':
        raise KeyError('Sorry, LSCV method have not implemented yet.')
    
    else:
        raise ValueError(f"method should be ['median', 'silverman', 'scott', 'LSCV']")
    
    return sigma


class gauss_kernel():
    def __init__(self,sigma):
        self.sigma = sigma
        
    def __call__(self,x,y):
        return self.compute(x,y)
    
    def compute(self, x,y):
        if x.shape != y.shape:
            raise ValueError(f"operands could not be broadcast together with shapes {x.shape} {y.shape}")
        
        p = scp.stats.multivariate_normal.pdf(x,y,self.sigma)
        return p
