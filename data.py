import csv
import numpy as np

def load(dataset):
	X = np.load('data/'+dataset+'.npy').astype(np.float32)

	if dataset == 'train':
		f = file('data/'+dataset+'.txt')
		r = csv.reader(f)
		y = np.array([float(row[0]) for row in r], np.float32)
		return X, y

	return X

def normalize(train, test):
	''' Normalize feature-wise (RGB). '''

	mean = np.mean(train,(0,1,2))
	std = np.std(train,(0,1,2))

	train -= mean
	train /= std

	test -= mean
	test /= std

def split(X, y):
	''' Split training data 80-20. '''
	splitter = len(X) / 5
	X_train = X[0:splitter]
	X_val = X[splitter:]

	y_train = y[0:splitter]
	y_val = X[splitter:]

	return X_train, X_val, y_train, y_val
