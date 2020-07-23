from ..KernelMean import KernelMean as Mean
import pandas as pd
import numpy as np

class KernelDataSet_for_ABC():
    def __init__(self, prior_samples, parameter_keys, observed_samples, data_key):
        if not isinstance(prior_samples, pd.DataFrame):
            raise TypeError(f'Type of prior_samples should be pandas Dataframe.')
        if not isinstance(observed_samples, pd.DataFrame):
            raise TypeError(f'Type of observed_samples should be pandas Dataframe.')
        if not (isinstance(parameter_keys, list) and isinstance(data_key, list)):
            raise TypeError(f'Type of keys should be list.')
        self.row_samples = prior_samples
        self.row_obs = observed_samples
        self.parameter_keys = parameter_keys
        self.data_key = data_key
        self._duplicate_parameter(prior_samples, observed_samples, parameter_keys, data_key)
    
    def reset_keys(self,parameter_keys=None, data_key=None):
        if data_key is None:
            data_key = self.data_key
        if parameter_keys is None:
            parameter_keys = self.parameter_keys
        if not (isinstance(parameter_keys, list) and isinstance(data_key, list)):
            raise TypeError(f'Type of keys should be list.')
        
        self._duplicate_parameter(self.row_samples, self.row_obs, parameter_keys, data_key)
        self.data_key = data_key
        self.parameter_keys = parameter_keys
        
    def _duplicate_parameter(self, 
                             prior_samples,
                             observed_samples,
                             parameter_keys,
                             data_key):
        duplicates_df = prior_samples[parameter_keys].drop_duplicates().reset_index(drop=True)
        duplicates_df.index.names = ['para_idx']
        self.parameters = duplicates_df
        
        self.observed_samples = observed_samples.loc[:,data_key].T
        
        tmp = pd.DataFrame()
        for i,row in duplicates_df.iterrows():
            tmp = pd.concat([tmp,
                             prior_samples[(prior_samples[parameter_keys]==row).all(axis=1)].reset_index(drop=True).reset_index().assign(para_idx=i)],
                            axis=0)
        self.prior_data = tmp.pivot(
            index='para_idx', 
            values=data_key, 
            columns='index')
