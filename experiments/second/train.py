import sys

from keras.models import Sequential
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import SimpleRNN

from container import ModelContainer

def construct():
	model = Sequential()
	model.add(TimeDistributed(ZeroPadding2D((3, 3)), input_shape=(None,120,160,3)))
	model.add(TimeDistributed(Convolution2D(32, 5, 5, activation='relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(2, 2))))

	model.add(TimeDistributed(ZeroPadding2D((3, 3))))
	model.add(TimeDistributed(Convolution2D(32, 5, 5, activation='relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(2, 2))))

	model.add(TimeDistributed(ZeroPadding2D((3, 3))))
	model.add(TimeDistributed(Convolution2D(64, 5, 5, activation='relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))

	model.add(TimeDistributed(ZeroPadding2D((3, 3))))
	model.add(TimeDistributed(Convolution2D(128, 5, 5, activation='relu')))
	model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))

	model.add(TimeDistributed(Flatten()))
	model.add(TimeDistributed(Dense(128)))
	model.add(SimpleRNN(128, return_sequences=True))
	model.add(TimeDistributed(Dense(1)))
	return model

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 4:
		print "Usage: train nb_epoch batch_size samples_per_epoch"
		sys.exit()

	model = ModelContainer(name,construct(),10,"adam")
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
