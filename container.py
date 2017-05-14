from data import load, normalize

class ModelContainer:
	def __init__(self,name,model,optimizer):
		X_train, y_train = load('train')
		X_test = load('test')
		normalize(X_train)

	def sample_gen(self):
		pass

	def train(self):
		pass

	def evaluate(self):
		pass

if __name__ == '__main__':
	m = ModelContainer(None,None,None)