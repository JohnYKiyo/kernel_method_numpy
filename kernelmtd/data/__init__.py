import pandas as pd
import numpy as np
import os
absolute_path = os.path.dirname(os.path.abspath(__file__))
testdata = pd.read_csv(absolute_path+'/test_data1.csv')
regression_data = np.load(absolute_path+'/regression.npy', allow_pickle=True).item()
