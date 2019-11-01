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
        self.sigma = sigma # not covariance matrix
            
    def __call__(self,x,y):
        return self.compute(x,y)
        
    def compute(self, x,y):
        if len(x.shape) ==1:
            x = np.array([x]) # convert [x,x] to [[x,x]]
        if len(y.shape) ==1:
            y = np.array([y]) # convert [y,y] to [[y,y]]
        
        dim = x.shape[1]
        log_pdf = -1*euclidean_distances(x,y)/(2*(self.sigma**2))
        val =  np.exp(log_pdf) /((2*np.pi*self.sigma**2)**(dim/2.))
        """
                               np.exp(log_pdf) 
            val  =  -----------------------------------------
                      ((2*np.pi*self.sigma**2)**(dim/2.))
        
        """
        #val = scp.stats.multivariate_normal.pdf(x,y,self.sigma)
        return val
