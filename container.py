import random, os, csv
from keras.callbacks import ModelCheckpoint
from data import *

class ModelContainer:
	def __init__(self,name,model,optimizer,callbacks=[]):
		self.name = name
		self.callbacks = callbacks
		random.seed(42)

		model.compile(optimizer=optimizer, loss="mse")
		self.model = model

		print "Loading train data..."
		X_train, y_train = load('train')

		print "Loading test data..."
		X_test = load('test')

		print "Stacking data..."
		X_train, y_train = stack(X_train, y_train)
		self.X_test = stack(X_test)

		self.X_train, self.X_val, self.y_train, self.y_val = split(X_train, y_train)

	def sample_gen(self, batch_size):
		X_batch = []
		y_batch = []

		while True:
			if len(X_batch) == batch_size:
				y_batch = np.array(y_batch)
				y_batch = y_batch.reshape((y_batch.shape[0],1))
				yield np.array(X_batch), y_batch

				X_batch = []
				y_batch = []

			sample = random.randrange(len(self.X_train))

			X_batch.append(self.X_train[sample])
			y_batch.append(self.y_train[sample])

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

		predictions = self.model.predict(self.X_test, verbose=True)[:,:,0]

		f = file('experiments/'+self.name+'/test.txt','wb')
		w = csv.writer(f)
		w.writerow([0.])
		[w.writerows([[e] for e in seq]) for seq in predictions]
		f.close()
		print "Done. Wrote experiments/"+self.name+"/test.txt."
