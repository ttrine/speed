import csv, math
import numpy as np

def load(dataset):
	X = np.load('data/'+dataset+'.npy').astype(np.float32)

	if dataset == 'train':
		f = file('data/'+dataset+'.txt')
		r = csv.reader(f)
		y = np.array([float(row[0]) for row in r], np.float32)
		return X, y

	return X

def split(X, y):
	''' Split training data 80-20. '''
	splitter = len(X) / 5

	X_val = X[0:splitter]
	X_train = X[splitter:]

	y_val = y[0:splitter]
	y_train = y[splitter:]

	return X_train, X_val, y_train, y_val

def stack(X, y=None):
	# Turn X from (samples, length, width, 3)
		# to (samples-1, length, width, 6)
	X_new = np.concatenate((X[1:],X[:-1]),3)

	# Remove first element of y
	if y is not None:
		y_new = y[1:]
		return X_new, y_new

	return X_new
