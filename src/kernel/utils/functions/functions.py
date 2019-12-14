from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import scipy as scp

def convert_array(x):
    if isinstance(x,list):
        x = np.array(x)
    
    if len(x.shape) ==1:
        x = np.array([x]) # convert [x_1,x_2] to [[x_1,x_2]] d is dimention
    return x

def get_band_width(d, method='median'):
    '''
    Determine bandwidth of gaussian kernel. 
    Method:
        - Median heuristic
        - Silverman heuristic
        - Scott heuristic
        - LSCV
    '''
    ndim = d.shape[1]    
    if method == 'median':
        K = euclidean_distances(d, squared=True) #データ間のユークリッド距離を総当たり的に求めてる
        sigma = np.median(np.sqrt(K)) #最頻する距離で正規化するためにmedian取ってる
        
    elif method in ['silverman', 'scott']:
        factor = 1.06 if method == 'silverman' else 1.0
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

    '''
    in future.
    H = diag(h_1,h_2,...,h_d)
    and \Sigma = diag(\sigma_1,...,\sigma_d)
    
    so, j-th band width is 
    h_j = (4/(j+2))**(1/(d+1)) * n**(-1/(d+1)) * sig_j
    ref: Wolfgang Karl et al., "Nonparametric and Semiparametric Models", 2012
    '''

class gauss_kernel():
    def __init__(self,sigma):
        self.sigma = sigma # not covariance matrix
       
    def __call__(self,x,y):
        return self.compute(x,y)

    def compute(self, x,y,normalize=True):
        x,y = convert_array(x), convert_array(x)

        nom_factor = 1
        if normalize:
            dim = x.shape[0]
            nom_factor = self._normalize_factor(dim)

        log_pdf = -1*euclidean_distances(x,y,squared=True)/(2*(self.sigma**2))
        val =  np.exp(log_pdf) / nom_factor
        """
                               np.exp(log_pdf) 
            val  =  -----------------------------------------
                      ((2*np.pi*self.sigma**2)**(dim/2.))
        
        """
        return val

    def _normalize_factor(self,d):
        return (2*np.pi*self.sigma**2)**(d/2.)
   
    def grad(self, x, y):
        if len(x.shape) ==1:
            x = np.array([x]) #convert [x,x,...,x] to [[x,x,...,x]
        if len(y.shape) ==1:
            y = np.array([y]) #convert [y,y,...,y] to [[y,y,...,y]]
    
        dim = x.shape[1]
        Nx_Samples = x.shape[0]
        Ny_Samples = y.shape[0]
    
        #compute kernels at each sample
        k = self.compute(x,y)
        #repeat kernels at each sample 'dim' times in order to calculate element wise product for gradient.
        V = np.reshape(np.tile(k,dim), (Nx_Samples, dim, Ny_Samples) )
        """
       
                                       np.exp(log_pdf) 
          V_{k,d,i}   =  -----------------------------------------
                            ((2*np.pi*self.sigma**2)**(dim/2.))
   
        """
    
        #compute element wise differences at each sample
        W = -1*(x.reshape(Nx_Samples, 1, dim) - y).transpose(0, 2, 1) / (self.sigma**2)
        """
                           - (x^d_k - y^d_i)
            W_{k,d,i} =  ---------------------
                               sigma**2
        """
        return W*V

class gram_matrix(gauss_kernel):
    def __init__(self,x,sigma):
        super().__init__(sigma)
        self.gram_matrix = self.compute(x,x,normalize=False)
        self.normalize_factor = self._normalize_factor(x.shape[0])

    def __call__(self):
        return self.gram_matrix 
