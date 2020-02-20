'''
misc.py

Created by 

name: Federico Marcuzzi
e-mail: federico.marcuzzi@unive.it

date 20/02/2020
'''

import numpy as np
import pandas as pd


# normalizes dataset in [0,1].
def normalize(dataset):
	dataset += np.min(dataset, axis=0)
	return dataset / (np.max(dataset, axis=0) + 1e-32)

# loads dataset from csv.
def load_dataset(path):
	file_name = path + ".csv"
	data = np.asarray(pd.read_csv(file_name,header=None))
	return data[:,:-1], data[:,-1]