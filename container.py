import random
from data import *

class ModelContainer:
	def __init__(self,name,model,n,optimizer):
		self.name = name
		self.model = model
		self.n = n

		# model.compile(optimizer=optimizer, loss="mse")

		X_train, y_train = load('train')
		X_test = load('test')
		normalize(X_train, self.X_test)

		X_train, X_val, y_train, y_val = split(X_train, y_train)

		self.X_val, self.y_val = sequence(n, X_val, y_val)
		self.X_test = sequence(n, X_test)

	def sample_gen(self, batch_size):
		X_batch = []
		y_batch = []

		while True:
			# if len(X_batch) == batch_size:
				# yield np.array(X_batch), np.array(y_batch)
				# X_batch = []
				# y_batch = []

			# range(len(self.X_train))
			# X_batch.append(X[])

			pass

	def train(self):
		pass

	def evaluate(self):
		# Break test set into length-n sequences
		# Run inference on each sequence in order
		# Flatten out the predictions from each sequence
		# Write to CSV
		pass

if __name__ == '__main__':
	m = ModelContainer(None,None,None,None)