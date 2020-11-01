import pandas as pd
import numpy as np
testdata = pd.read_csv('./kernelmtd/data/test_data1.csv')
regression_data = np.load('./kernelmtd/data/regression.npy', allow_pickle=True).item()
