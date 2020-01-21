import numpy as np
import scipy as scp
class Hamiltonian():
    '''
    Parameters:
    '''
    def __init__(self, logp, grad_logp, ndim, initial_position, T, eps, tau):
        self._logp = logp #potential energy
        self._dHdq = grad_logp

        self._ndim = ndim
        self._T = T
        self._eps = eps
        self._tau = tau

        self._init_position = np.atleast_1d(initial_position)

    def _momentum(self,p):
        return np.square(p).sum()/(2*self._tau**2)

    def _d_momentum(self,p):
        return p/(self._tau**2)

    def _hamiltonian(self,q,p):
        return self._momentum(p) -1.*self._logp(q)

    def _leapfrog(self, q, p):
        """
        Parameters
        q: np.float  Initial position.
        p: np.float  Initial momentum.
        dVdq:        Gradient of the hamiltonian.
        path_len:    integrate length.
        step_size:   each integration step.

        q_{t+1} = q_{t}+dq/dt * dt
        p_{t+1} = p_{t}+dp/dt * dt
        """
        q = np.atleast_1d(q)
        q += -0.5*self._eps*(-1.*self._d_momentum(p))
        p += self._eps*self._dHdq(q)
        q += -self._eps*(-1.*self._d_momentum(p))

        return q, p

    def _step_HMC_iter(self,q):
        p = np.random.normal(0,self._tau,size=self._ndim)
        q_new, p_new = np.copy(q), np.copy(p)
        for t in range(self._T):
            q_new, p_new = self._leapfrog(q_new,p_new)

        alpha = np.exp(self._hamiltonian(q_new,p_new) - self._hamiltonian(q_new,p_new))
        u = np.random.uniform()
        if u < alpha:
            q_accept = q_new
        else:
            q_accept = q
        return q_accept

    def sampling(self,n_samples,reject=0):
        samples = [self._init_position]
        for i in range(n_samples):
            samples.append(self._step_HMC_iter(samples[i]))
        return np.array(samples[1+reject:])