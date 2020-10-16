#import kernel
#import simulation_model

#reload(kernel)
#reload(simulation_model)

import scipy.optimize as so
import numpy as np
import scipy.linalg as linalg 
from sklearn.metrics.pairwise import euclidean_distances

class kernelabc:
    
    def __init__(self, theta_pri, y_obs, y_sim, sigma_y=0., epsilon_const=0.01):
#        self.k = kernel.kernel()
        
        self.theta_pri = theta_pri
        self.n_theta_sample, self.n_theta_dim = theta_pri.shape
        self.n_obs = y_obs.shape[0]
        
        self.y_obs = y_obs
        self.y_sim = y_sim
        
        self.ky = np.zeros(self.n_theta_sample)

        self.Gy = np.zeros((self.n_theta_sample, self.n_theta_sample))
        self.w = np.zeros(self.n_theta_sample)
        self.sigma_y = sigma_y
        
        self.epsilon_const = epsilon_const
        self.epsilon = self.epsilon_const / np.sqrt(self.n_theta_sample)
        
        self.post_mean = np.zeros((self.n_theta_dim))
        self.theta_herding = np.zeros(self.theta_pri.shape)
        
        self.Gt = np.zeros((self.n_theta_sample, self.n_theta_sample))
        self.Gtt = np.zeros((self.n_theta_sample, self.n_theta_sample))
        
        self.v = np.zeros(self.n_theta_sample)
        self.ky_pred = np.zeros(self.n_theta_sample)
        
        self.pred_mean = np.zeros(1)
        
        
    def gaussian_kernel(self, x, y, sigma):
        k = np.exp( -(np.linalg.norm(x - y)**2) / ( 2*sigma*sigma))
        return k
    
    
    def compute_gramian_fast(self, data, sigma):

        K = euclidean_distances(data, squared=True)
        K *= -1.0 /(2*sigma*sigma) 
        gramian = np.exp(K, K)

        return gramian
    
    
    def compute_median(self, data):
        K = euclidean_distances(data, squared=True)
        sigma = np.median(np.sqrt(K))        
        return sigma
    
    
    def kernel_herding(self, x, y, y_sq, sigma):
        const1 = 1.
        if y.shape[1]==0:
            return 0
        return 1.0/const1*np.exp(-1./(2.*sigma**2)*(sum(x ** 2) - 2.*x.dot(y) + y_sq))
    
        
    def regression(self, output_result=False):
   
        ### weight for each point in X space
        beta_obs = self.beta(self.X_obs)

        ### simulate for each theta
        y_sim_list = []
        for theta_i in self.theta_pri:
            y_sim_list.append(self.simulator.simulate(self.X_obs, theta_i))            
        self.y_sim = np.nan_to_num(y_sim_list)

        ### weighted data for sim and obs !!!difference from regression function is only here !!! 
        y_sim_weighted = np.zeros((self.n_theta_sample, self.n_obs))
        y_obs_weighted = np.zeros(self.n_obs)
        for t in range(self.n_obs): # TODO: should be matrix form
            y_sim_weighted[:,t] = beta_obs[t]*self.y_sim[:,t]
            y_obs_weighted[t] = beta_obs[t]*self.y_obs[t]

        ### calcurate G:=(ky(y,y)), w, ky
        sigma_y_median = self.compute_median(y_sim_weighted)
        print(f"sigma_y_median:{sigma_y_median}")
        if self.sigma_y == 0.:
            self.sigma_y = sigma_y_median
        self.Gy = self.compute_gramian_fast(y_sim_weighted, self.sigma_y)        
        print(f"self.sigma_y:{self.sigma_y}") #debug
#        self.Gy, self.sigma_y = self.k.compute_gramian_fast(y_sim_weighted)
    
        ### Eq(3) at Fukumizu2013
        for i in nrange(self.n_theta_sample): #step output
            self.ky[i] = self.gaussian_kernel(y_sim_weighted[i,:], y_obs_weighted, self.sigma_y)
        self.w = np.dot(np.linalg.inv(self.Gy + self.n_theta_sample * self.epsilon * np.eye(self.n_theta_sample)), self.ky)# Eq(19) at Fukumizu2013
    
        # calcurate the posterior mean
        self.post_mean = np.round(np.average(self.theta_pri, axis=0, weights=self.w), 2) # the mean of the posterior distribution
        print(self.post_mean)
    
        if output_result:
            return self.post_mean, self.Gy, self.w, self.sigma_y, self.ky
    

    def supersample(self, sigma_herding=1.,N=None):
        """
        comment
        """
        if N is None:
            N = self.n_theta_sample
        # transverse of input theta prior
        X = self.theta_pri.T # shape: (n_theta_dim, n_theta_sample)
        # ouput super sampled theta (transverse)
        S = np.zeros([self.n_theta_dim, N])

        # turning paramter of mixed kernel
        # inner of contant kernel (?) calulated from input theta
        X_sq = (X**2).sum(axis=0)
        # inner of contant kernel (?) calulated from output theta
        S_sq = np.zeros(N)
        
        mu_p = np.dot(self.Gy + self.n_theta_sample * self.epsilon * np.eye(self.n_theta_sample), self.w)
        print(f"mu_p.shape:{mu_p.shape}") # debug
                
        # initialize state-space h_t
        
        h_0 = mu_p
        h_t_sum = np.zeros(N)
        
        ### loop of t = 0, ..., n_theta_sample
        
        for t in range(N):
        
            if t > 0:
                h_t_sum = h_t_sum + self.kernel_herding(S[:, t-1], X, X_sq, sigma_herding) #TODO  be gramian?                                                    
            h_t = (t+1)*h_0 - h_t_sum
            idx = np.argmax(h_t)
            S0 = X[:,idx]
            
            # sampling to maximize Eq.4 as dual formulation in Chen2010
            if t==0:
                g_t = lambda x: -np.dot(self.kernel_herding(x,X,X_sq,sigma_herding), self.w)
            else: 
                g_t = lambda x: -((t+1)*np.dot(self.kernel_herding(x,X,X_sq,sigma_herding), self.w) - sum(self.kernel_herding(x,S[:,:t],S_sq[:t],sigma_herding)))

            res = so.minimize(g_t, S0, method="TNC") # Turncated Newton algorithm so.minimize(obj_func, init_state, method)
            S[:,t] = res.x #
            S_sq[t] = np.dot(S[:,t], S[:,t])

        self.theta_herding = S.T
        return self.theta_herding

    
    
    def density_estimation(self, output_result=False):

        ### calcurate G:=(ky(y,y)), w, ky
        sigma_y_median = self.compute_median(self.y_sim)
#        print "sigma_y_median: ", sigma_y_median
        if self.sigma_y == 0.:
            self.sigma_y = sigma_y_median
        self.Gy = self.compute_gramian_fast(self.y_sim, self.sigma_y)
#        print "self.sigma_y: ", self.sigma_y #debug    

        ### Eq(3) at Fukumizu2013
        for i in range(self.n_theta_sample): #step output
            self.ky[i] = self.gaussian_kernel(self.y_sim[i,:], self.y_obs, self.sigma_y*3.5)
        self.w = np.dot(np.linalg.inv(self.Gy + self.n_theta_sample * self.epsilon * np.eye(self.n_theta_sample)), self.ky)# Eq(19) at Fukumizu2013
    
        # calcurate the posterior mean
        self.post_mean = np.round(np.average(self.theta_pri, axis=0, weights=self.w), 2) # the mean of the posterior distribution
    
        if output_result:
            return self.post_mean, self.Gy, self.w, self.sigma_y, self.ky