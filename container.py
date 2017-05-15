import random, os
from keras.callbacks import ModelCheckpoint
from data import *

class ModelContainer:
	def __init__(self,name,model,n,optimizer,callbacks=[]):
		self.name = name
		self.n = n
		self.callbacks = callbacks
		random.seed(42)

		model.compile(optimizer=optimizer, loss="mse")
		self.model = model

		print "Loading train data..."
		X_train, y_train = load('train')
		print "Loading test data..."
		X_test = load('test')

		X_train, X_val, y_train, y_val = split(X_train, y_train)
		self.X_train = X_train
		self.y_train = y_train

		print "Sequencing data..."
		self.X_val, self.y_val = sequence(n, X_val, y_val)
		self.X_test = sequence(n, X_test)

	def sample_gen(self, batch_size):
		X_batch = []
		y_batch = []

		while True:
			if len(X_batch) == batch_size:
				y_batch = np.array(y_batch)
				y_batch = y_batch.reshape((y_batch.shape[0],y_batch.shape[1],1))
				yield np.array(X_batch), y_batch
				X_batch = []
				y_batch = []

			start = random.randrange(len(self.X_train) - self.n)

			X_batch.append(self.X_train[start:start + self.n])
			y_batch.append(self.y_train[start:start + self.n])

	def train(self, weight_file=None, nb_epoch=40, batch_size=500, samples_per_epoch=10000):
		model_folder = 'experiments/' + self.name + '/weights/'
		if not os.path.exists(model_folder):
			os.makedirs(model_folder)

		if weight_file is not None:
			self.model.load_weights(model_folder + self.name + weight_file)
		
		model_checkpoint = ModelCheckpoint(model_folder+'{epoch:002d}-{val_loss:.4f}.hdf5', monitor='loss')
		self.callbacks.append(model_checkpoint)
		train_gen = self.sample_gen(batch_size)

		print "Running training..."
		self.model.fit_generator(train_gen, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, 
			validation_data=(self.X_val,self.y_val), verbose=1, callbacks=self.callbacks)

	def evaluate(self,weight_file):
		self.model.load_weights('experiments/'+self.name+'/weights/'+weight_file)

		print "Running inference..."

		predictions = self.model.predict(self.X_test, verbose=True)

		# f = file('experiments/'+self.name+'/test.txt','wb')
		# w = csv.writer(f)
		# w.writerows()
		# f.close()
		# print "Done. Wrote experiments/"+self.name+"/test.txt."
