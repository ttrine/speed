import sys

from keras.models import Model
from keras.layers import Input, merge, AveragePooling2D, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from container import ModelContainer

# Inception module where each 5x5 convolution is 
	# instead factored into two 3x3 convolutions.
def factor_5x5(x,nb_1x1,nb_3x3_reduce,nb_3x3,nb_3x3dbl_reduce,nb_3x3dbl):
	branch1x1 = Convolution2D(nb_1x1, 1, 1, border_mode='same', activation='relu')(x)

	branch3x3 = Convolution2D(nb_3x3_reduce, 1, 1, border_mode='same', activation='relu')(x)
	branch3x3 = Convolution2D(nb_3x3, 3, 3, border_mode='same', activation='relu')(branch3x3)

	branch3x3dbl = Convolution2D(nb_3x3dbl_reduce, 1, 1, border_mode='same', activation='relu')(x)
	branch3x3dbl = Convolution2D(nb_3x3dbl, 3, 3, border_mode='same', activation='relu')(branch3x3dbl)
	branch3x3dbl = Convolution2D(nb_3x3dbl, 3, 3, border_mode='same', activation='relu')(branch3x3dbl)

	x = merge([branch1x1, branch3x3, branch3x3dbl], mode='concat')

	return x

# Same as factor_5x5, but also pools the output 
	# to half the size.
def pool_5x5(x,nb_1x1,nb_3x3_reduce,nb_3x3,nb_3x3dbl_reduce,nb_3x3dbl,nb_pool):
	branch1x1 = ZeroPadding2D((0,1))(x)
	branch1x1 = Convolution2D(nb_1x1, 1, 1, subsample=(2,2), border_mode='same', activation='relu')(branch1x1)

	branch3x3 = ZeroPadding2D((0,1))(x)
	branch3x3 = Convolution2D(nb_3x3_reduce, 1, 1, border_mode='same', activation='relu')(branch3x3)
	branch3x3 = Convolution2D(nb_3x3, 3, 3, subsample=(2,2), border_mode='same', activation='relu')(branch3x3)

	branch3x3dbl = ZeroPadding2D((0,1))(x)
	branch3x3dbl = Convolution2D(nb_3x3dbl_reduce, 1, 1, border_mode='same', activation='relu')(branch3x3dbl)
	branch3x3dbl = Convolution2D(nb_3x3dbl, 3, 3, border_mode='same', activation='relu')(branch3x3dbl)
	branch3x3dbl = Convolution2D(nb_3x3dbl, 3, 3, subsample=(2,2), border_mode='same', activation='relu')(branch3x3dbl)

	branch_pool = ZeroPadding2D((0,1))(x)
	branch_pool = AveragePooling2D((3, 3), 
					strides=(2,2), border_mode='same')(branch_pool)
	branch_pool = Convolution2D(nb_pool, 1, 1)(branch_pool)

	x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool], mode='concat')

	return x

def construct():
	inp = Input(shape=(120,160,6))
	
	x = Convolution2D(16, 3, 3, subsample=(2,2), border_mode="same", activation="relu")(inp)
	x = Convolution2D(32, 3, 3, border_mode="same", activation="relu")(x)
	x = MaxPooling2D((3,3), strides=(2,2))(x)
	x = Convolution2D(64, 3, 3, subsample=(2,2), border_mode="same", activation="relu")(x)
	x = Convolution2D(64, 3, 3, subsample=(2,2), border_mode="same", activation="relu")(x)
	x = BatchNormalization()(x)

	x = factor_5x5(x,40,24,40,32,48)
	x = pool_5x5(x,64,48,64,64,96,32)
	x = BatchNormalization()(x)

	x = factor_5x5(x,64,48,64,64,64)
	x = pool_5x5(x,64,48,64,64,64,32)
	x = BatchNormalization()(x)

	x = pool_5x5(x,70,64,70,55,96,43)
	x = pool_5x5(x,96,80,96,96,96,54)
	x = pool_5x5(x,96,96,96,96,192,64)
	x = BatchNormalization()(x)

	fcn = Flatten()(x)
	fcn = Dropout(.6)(fcn)
	fcn = Dense(128)(fcn)
	fcn = Dense(1)(fcn)

	return Model(input=inp,output=fcn)

datagen_args = dict(rotation_range=5.,
					width_shift_range=0.07,
					height_shift_range=0.07,
					zoom_range=.07,
					channel_shift_range=0.,
					horizontal_flip=True)

if __name__ == '__main__':
	import sys # basic arg parsing, infer name

	if len(sys.argv) < 4:
		print "Usage: train nb_epoch batch_size samples_per_epoch"
		sys.exit()

	name = sys.argv[0].split('/')[-2]
	
	model = ModelContainer(name,construct(),"adam",datagen_args=datagen_args)
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
