import csv, math, random
import numpy as np

random.seed(42)

def load(dataset):
	X = np.load('data/'+dataset+'.npy')

	if dataset == 'train':
		f = file('data/'+dataset+'.txt')
		r = csv.reader(f)
		y = np.array([float(row[0]) for row in r], np.float32)
		return X, y

	return X

def split(X, y):
	''' Set aside 20% of the data for training. 
		Data are partitioned into contiguous subregions 
		to reduce validation overfitting.'''
	while True:
		s = [[e, e+408] for e in random.sample(range(len(X)),10)]
		s.sort(key=lambda x: x[0])
		if s[-1][1] >= 20400: # Check boundary condition
			continue
		if all([s[i][1] < s[i+1][0] for i in range(len(s)-1)]): # Check non-overlapping condition
			break
		else: continue
		break

	val_inds = sum([range(e[0],e[1]) for e in s],[])
	train_inds = [e for e in range(len(X)) if e not in val_inds]
	return X[train_inds], X[val_inds], y[train_inds], y[val_inds]

def stack(X, y=None):
	# Turn X from (samples, length, width, 3)
		# to (samples-1, length, width, 6)
	X_new = np.concatenate((X[1:],X[:-1]),3)

	# Remove first element of y
	if y is not None:
		y_new = y[1:]
		return X_new, y_new

	return X_new
