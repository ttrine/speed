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
	X_train = X[0:splitter]
	X_val = X[splitter:]

	y_train = y[0:splitter]
	y_val = y[splitter:]

	return X_train, X_val, y_train, y_val

def sequence(n, X, y=None):
	# Turn X from (samples, length, width, 3)
		# to (samples/n, n, length, width, 3)
	num_seqs = int(math.ceil(X.shape[0] / float(n)) - 1)
	X_new = np.zeros((num_seqs,n,X.shape[1],X.shape[2],X.shape[3]))

	if y is not None:
		y_new = np.zeros((num_seqs,n))

	for i, ind in enumerate(range(n,len(X),n)):
		X_new[i] = X[ind - n:ind]
		if y is not None:
			y_new[i] = y[ind - n:ind]

	if y is not None:
		y_new = y_new.reshape((num_seqs,n,1))
		return X_new, y_new

	return X_new
