# band_width
**kernel.utils.functions.band_width**    
*class kernel.utils.functions.band_width(data, method=None, weights=None)*

**Parameters:**   

method: str
>method to calculate bandwidth   
 - Median heuristic   
 - Scott heuristic   
 - Silverman heuristic   
 - LSCV   
 - LCV   

data: array_like    
>Datapoints to estimate.   
1D-array: [x_1,x_2,...,x_n]   
1D-array input is treated as 1-dimensional data: [[x_1],[x_2],...,[x_n]]     
2D-array: [[x_1^d1,x_1^d2], [x_2^d1,x_2^d2],...,[x_n^d1,x_n^d2]]   

weights: array_like   
>weights of datapoints. This must be the same shape as dataset.   
 
**Attributes**   
cov:   
>The covariance matrix of dataset, scaled by the calculated bandwidth. 
 
inv_cov:   
>The inverse of covariance.

# gauss_kernel
**kernel.utils.functions.gauss_kernel**    
*class kernel.utils.functions.gauss_kernel(covariance, n_features)*

**Parameters:**    

covariance: array_like
>2D-array: Covariance matrix <img src="https://latex.codecogs.com/gif.latex?\Sigma"/>   
The covariance matrix of dataset, scaled by the calculated bandwidth.   
 
n_features: int   
>dimention of data. <img src="https://latex.codecogs.com/gif.latex?d"/>
 
**Attributes:**    
cov:   
>The covariance matrix of dataset, scaled by the calculated bandwidth. <img src="https://latex.codecogs.com/gif.latex?\Sigma"/>   

inv_cov:
>The inverse of covariance: <img src="https://latex.codecogs.com/gif.latex?Q"/>

*pdf(x,y,normalize=False)*:   
>probability density function estimated by gauss kernel functions.  
if normalize is False, normalize factor = 1.   
<img src="https://latex.codecogs.com/gif.latex?pdf(x,y)=\frac{\exp\{-0.5(x-y)^T&space;Q&space;(x-y)\}}{\sqrt{2\pi^d|\Sigma|}}"/>   
>x,y: 2D-array: (N_x , N_features) matrix or
>1D-array: [x_1,x_2,...,x_n] input is treated as 1-dimensional data: [[x_1],[x_2],...,[x_n]]     
>*return*: (N_x , N_y) matrix.   

*logpdf(x,y,normalize=False)*:   
>ln(pdf)    
<img src="https://latex.codecogs.com/gif.latex?logpdf(x,y)=-0.5(x-y)^T&space;Q&space;(x-y)-\log\left(\sqrt{2\pi^d|\Sigma|}\right)"/>   

norm_factor: 
> normalized factor of pdf.  <img src="https://latex.codecogs.com/gif.latex?\sqrt{2\pi^d|\Sigma|}"/>   

*grad(x,y,normalize=False)*:  
> gradient of pdf(x,y).

*grad_logpdf(x,y)*:   
> gradient of logpdf(x,y)
