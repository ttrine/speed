import csv
import numpy as np

def load_data(dataset):
	X = np.load('data/'+dataset+'.npy').astype(np.float32)

	if dataset == 'train':
		f = file('data/'+dataset+'.txt')
		r = csv.reader(f)
		y = np.array([float(row[0]) for row in r], np.float32)
		return X, y

	return X

def normalize(X):
	pass

if __name__ == '__main__':
	X_train, y_train = load_data('train')
	X_test = load_data('test')
	normalize(X_train)
