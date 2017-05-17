import csv, math
import numpy as np

from sklearn.model_selection import train_test_split

def load(dataset):
	X = np.load('data/'+dataset+'.npy')

	if dataset == 'train':
		f = file('data/'+dataset+'.txt')
		r = csv.reader(f)
		y = np.array([float(row[0]) for row in r], np.float32)
		return X, y

	return X

def split(X, y):
	''' Set aside 20% of the data for training. '''
	return train_test_split(
		X, y, test_size=0.2, random_state=42)

def stack(X, y=None):
	# Turn X from (samples, length, width, 3)
		# to (samples-1, length, width, 6)
	X_new = np.concatenate((X[1:],X[:-1]),3)

	# Remove first element of y
	if y is not None:
		y_new = y[1:]
		return X_new, y_new

	return X_new
