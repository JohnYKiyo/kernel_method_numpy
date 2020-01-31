from ...utils.progressbar import progbar
import numpy as np

class Metropolis():
    '''
    Sampling from a certain probability distribution $ p (x) = \ frac {\ hat {p} (x)} {Z_p} $.
    The normalization constant satisfies $ Z_p = \ int \ hat {p} (x) dx $.
    Samples from $ p (x) $ even if $ Z_p $ is not known by the Metropolis method.
    Prepare a distribution that can be directly sampled, called a proposed distribution,
    algorithm
    1. Initial value $ x_1 $
    2. One sample ($ x ^ * $) from the proposal distribution around $ x_i $
    3. $ min (1, \ frac {\ hat {p} (x ^ *)} {\ hat {p} (x_i)}) $ with a probability of $ x_ {i + 1} = x_i $
    4. The sequence $ \ {x_n \} _ {n = 1} ^ {N} $ obtained by repeating Steps 2 and 3 
       is taken as a sample from the probability distribution $ p (x) $.

    ndim : sample's dimention
    proposal_std : proposal distribution std
    sample_rate : Interval for thinning samples
    T : Number of ignore initial samples
    '''
    def __init__(self, f, ndim, proposal_std=1., sample_rate=1):
        self.f = f
        self.ndim = ndim
        self.proposal_std = proposal_std
        self.sample_rate = sample_rate
    
    def __call__(self,N,WarmUp=0):
        return self.sampling(N,WarmUp)
    
    def sampling(self,sample_size, WarmUp=0):
        x = np.zeros(self.ndim)
        samples = []
        for i in range(sample_size*self.sample_rate):
            #sampling from proposal distribution
            x_star = np.random.normal(loc=0.,
                                 scale=self.proposal_std, 
                                 size=self.ndim)
            x_star += x
            
            #accept probability
            accept_p = self.f(x_star)/self.f(x)
            if accept_p > np.random.uniform():
                x = x_star
                
            if i % self.sample_rate == 0:
                samples.append(x)
        return np.array(samples[WarmUp:])
