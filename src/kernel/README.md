# kernelMean
**kernel.Mean**    
*class kernel.Mean( data, cov, weights=None)*

**Parameters:**   

data: numpy.array(2D-array) or pandas dataframe   
>2D-array or dataframe: (N,d) matrix   
<img src="https://latex.codecogs.com/gif.latex?[[x_1^1,...,x_1^d\],...,[x_N^1,...,x_N^d]]"/>   

cov:   
>The covariance matrix of dataset, scaled by the calculated bandwidth.   

weights: array_like   
>weights of datapoints. This must be the same shape as dataset.   
Kernel mean is averaged by this weight.


**Attributes**   
data:   
>Data points for kernel calculation    
 
kernel: *class kernel.utils.functions.gauss_kernel*   
>kernel function. 

weights:   
>weights of datapoints.   
if weights is None,   
<img src="https://latex.codecogs.com/gif.latex?w=1/N"/>

**Method**   
*kernel_mean(self,val,normalize=False)*
>return value of kernel mean distribution.   
<img src="https://latex.codecogs.com/gif.latex?\sum_{i=0}^N w_i k(val,data_i)"/>   
normalize: Whether to use a normalized kernel.

# kernelHerding
**kernel.Herding**    
*class kernel.sampling.KernelHerding( object KernelMean)*

**Parameters:**   

Kernel Mean: class object kernel.[Mean](#kernelMean)   

**Attributes**   
KernelMean:   class kernel.[Mean](#kernelMean)
> kernel mean for kernel herding.  

samples:   
>Herding samples (supersamples).

**Method**   
*supersample(self, sample_size, verbose={'clear_display':True, 'display':'notebook'}, optimizer='scipy_optim', \*\*args)*   
>compute kernel herding.   
Efficient resampling method from kernel mean.   
[[1]](https://arxiv.org/abs/1203.3472) Y. Chen, M. Welling, and A. Smola, “Super-samples from kernel herding,” in Proceedings of the 26th Conference on Uncertainty in Artificial Intelligence, UAI 2010, 2010, pp. 109–116.   

> sample_size: int    
sample size of herding.

> vervose: dict
clear_display: {True | False}   
Overwrite progress bar when updating herding progress bar.    
display: {'notebook' | 'command'}   
Execution environment is ipython notebook or command line.

> optimizer: {'scipy_optim'|'optuna'}   

> args:   
see scipy.optimizer.[minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)   
see optuna.study.[optimize](https://optuna.readthedocs.io/en/latest/reference/study.html)   

>derivatives: {True|False}
Support for differentiation of kernel function to be used by **scipy** optimizer   

# kernelABC
**kernel.ABC**    
*class kernel.KernelABC(self,Dataset,cov_y=None,cov_para=None)*   

**Parameters:**   
Dataset:   
>Dataset for KernelABC. see [here](#KernelDataSet_for_ABC)   
cov_y:   
>The covariance matrix of prior data.   
cov_para:   
>The covariance matrix of prior data.   

**Attributes**   
Dataset:   
n_para_set:   
cov_y:   
cov_para:   
kernel_y:   
posterior_kernel:   
weights:   

**Method**   
posterior_kernel_mean(self, data):
posterior_mean(self):

# KernelDataSet_for_ABC
**kernel.utils.ABCDataSet**
*class kernel.utils.KernelDataSet.KernelDataSet_for_ABC(self, prior_samples, parameter_keys, observed_samples, data_key)*

**Parameters:**
prior_samples:   
| x | y |para0|para1|   
|:---|:---|:---|:---|   
|0.0|0.0|5.0|11.0|   
|1.0|11.0|5.0|11.0|   
|2.0|11.0|5.0|11.0|   
|3.0|11.0|5.0|11.0|   
|4.0|22.0|5.0|11.0|   

parameter_keys:   
>Select parameters to use for ABC.

observed_samples:   
| x | y |   
|:---|:---|   
|0.0|0.0|   
|1.0|11.0|   
|2.0|11.0|   
|3.0|11.0|   
|4.0|22.0|   

data_key:   
Select observed key to use for ABC

**Attributes**    
self.row_samples   
self.row_obs   
self.parameter_keys   
self.data_key   
self.parameters:   
>duplicated parameters
self.observed_samples   
>Observed data formatted based on key information.
self.prior_data   
>Prior parameter data formatted based on key information.