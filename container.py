from data import load, normalize, split

class ModelContainer:
	def __init__(self,name,model,n,optimizer):
		self.name = name
		self.model = model
		self.n = n

		# model.compile(optimizer=optimizer, loss="mse")

		X_train, y_train = load('train')
		self.X_test = load('test')
		normalize(X_train, self.X_test)

		X_train, X_val, y_train, y_val = split(X_train, y_train)

	def sample_gen(self):
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