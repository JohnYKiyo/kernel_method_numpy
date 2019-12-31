from ..utils.progressbar import progbar
from ..utils.functions import gauss_kernel
from ..KernelMean import KernelMean

import scipy as scp
import numpy as np
import optuna

class KernelHerding():
    def __init__(self, obj_KernelMean):
        if not isinstance(obj_KernelMean, KernelMean):
             raise TypeError(f"init object shuld be KernelMean class, but got {type(obj_KernelMean)}")

        self.KernelMean = obj_KernelMean
        self._kernel = obj_KernelMean.kernel
        self._samples = None
        
    def supersample(self, 
                    sample_size, 
                    verbose={'clear_display':True, 'display':'notebook'},
                    optimizer='scipy_optim',
                    **args):
        '''
        sample_size: sample size of herding.
        verbose:
            clear_display: {True | False}
                Overwrite progress bar when updating herding progress bar.
            display: {'notebook' | 'command'}
                Execution environment is ipython notebook or command line.
        optimizer: {'scipy_optim'|'optuna'}
        args:
            see scipy.optimizer.minimize args
        '''
        self._samples = None

        prob = progbar(sample_size, **verbose)
        h = self.KernelMean.kernel_mean
        x1 = self._argmax(h, optimizer, **args)
        samples = np.atleast_2d(x1)
        for idx in range(sample_size-1):
            h = self._herding_update(samples)
            x = self._argmax(h, optimizer, **args)
            samples = np.append(samples,np.array([x]),axis=0)
            prob.update(idx+1)
            self._samples = samples

        del prob
        return samples

    def _argmax(self, h, optimizer, **args):
        minus_h = lambda x: -1.*h(x)

        if optimizer =='scipy_optim':
            optimize_result = self._optimizer_scipy(minus_h,**args)
            return optimize_result.x.squeeze()

        if optimizer =='optuna':
            optimize_result = self._optimizer_TPE(minus_h, **args)
            return np.array(list(optimize_result.best_params.values()))

    def _herding_update(self, samples):
        f = lambda x: self.KernelMean.kernel_mean(x) - np.average(self._kernel.pdf(x,samples),axis=1)
        return f
    
    def supersamples_kernel_mean(self,x):
        #         , sigma='median'):
        #         if isinstance(sigma,str):
        #             sigma = get_band_width(self._samples, sigma)
        #         kernel = gauss_kernel(sigma)
        return np.average(self._kernel.pdf(x,self._samples),axis=1)
    
    def _kernel_gradient_from_samples(self,x,samples):
        return np.average(self._kernel.grad(x,samples), axis=1).squeeze()
    
    def _optimizer_scipy(self, h, derivatives=False,**args):
        if 'x0' not in args:
            args['x0'] = self.KernelMean.data.sample().values
        if 'method' not in args:
            args['method'] = 'Powell'

        if derivatives:
            if self._samples is not None:
                args['jac'] = lambda x: -1*(self._kernel_gradient_from_samples(x, self.KernelMean.data.values) - self._kernel_gradient_from_samples(x,self._samples))
            else:
                args['jac'] = lambda x: -1*(self._kernel_gradient_from_samples(x, self.KernelMean.data.values))
            #else:
            #    raise ValueError(f'derivatives is not defined. The gradient is needed when using [CG, BFGS, l-bfgs-b,tnc, slsqp]')
 
        mins = (self.KernelMean.data.min() - 3*np.diag(self._kernel.cov)).values  ## in the future, np.diag(sigma)
        maxs = (self.KernelMean.data.max() + 3*np.diag(self._kernel.cov)).values  ## in the future, np.diag(sigma)
        optimize_fail = True
        while optimize_fail:
            optimize_result = scp.optimize.minimize(h, **args)
            if not optimize_result.success or (optimize_result.x < mins).any() or (optimize_result.x > maxs).any():
                args['x0'] = self.KernelMean.data.sample().values
                continue

            optimize_fail = False
        return optimize_result

    def _optimizer_TPE(self, h, **args):
        dim = self.KernelMean.data.shape[1]
        #min max of feature values
        mins = (self.KernelMean.data.min() - 3*np.diag(self._kernel.cov)).values  ## in the future, np.diag(sigma)
        maxs = (self.KernelMean.data.max() + 3*np.diag(self._kernel.cov)).values  ## in the future, np.diag(sigma)

        if 'n_trials' not in args:
            args['n_trials'] = 100
        if 'f_trials' not in args:
            args['f_trials'] = ['uniform']*dim      # ['uniform' for i in range(dim)]
        else:
            if len(args['f_trials']) != dim:
                length = len(args['f_trials'])
                raise TypeError(f'The size of trials must be the same size as the feature dimension, but given {length}')

        def objective(trial):
            f = []
            for i in range(dim):
                if args['f_trials'][i] == 'uniform':
                    f.append(trial.suggest_uniform(f'f{i}', mins[i], maxs[i]))

                elif args['f_trials'][i] == 'loguniform':
                    f.append(trial.suggest_loguniform(f'f{i}', mins[i], maxs[i]))

                else:
                    raise ValueError('f_trials should be \{\'uniform\' | \'loguniform\'\}')

            return h(np.array(f))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study()
        study.optimize(objective, n_trials=args['n_trials'])
        return study

    @property
    def samples(self):
        if self._samples is None:
            raise ValueError(f'super samples do not exist.')
        return self._samples