import sys

from keras.models import Sequential
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from container import ModelContainer

def construct():
	model = Sequential()
	model.add(BatchNormalization(input_shape=(120,160,6)))
	model.add(ZeroPadding2D((3, 3)))
	model.add(Convolution2D(32, 5, 5, activation='relu'))
	model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))

	model.add(ZeroPadding2D((3, 3)))
	model.add(Convolution2D(32, 5, 5, activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64))
	model.add(Dense(1))

	return model

if __name__ == '__main__':
	import sys # basic arg parsing, infer name

	if len(sys.argv) < 4:
		print "Usage: train nb_epoch batch_size samples_per_epoch"
		sys.exit()

	name = sys.argv[0].split('/')[-2]
	
	model = ModelContainer(name,construct(),"adam")
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
