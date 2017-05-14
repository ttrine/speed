import sys
sys.setrecursionlimit(10000)

import numpy as np

from keras import backend as K
from keras.models import Model

from keras.layers import Input, Dense, Flatten, merge, Reshape, ZeroPadding2D, Convolution2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda, Dropout, Activation, SpatialDropout2D
from keras.layers.core import Masking, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

from keras.callbacks import Callback

from keras.regularizers import l2

from fish.layers import SpecialBatchNormalization
from fish.classify import ClassifierContainer

# Lambda function for a differentiable layer that regularizes
	# classifier input according to the detector.
def fishy_features(x, tied_to):
	weights = tied_to.weights[0].reshape((1,1,1,256))
	fish_repeat = np.repeat(np.repeat(np.repeat(weights,x.shape[0],0),16,1),28,2)
	return x * fish_repeat

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

# Stem. Shared CNN for learning representations 
# 		common to detection and classification.
def stem(x):
	x = SpecialBatchNormalization()(x)
	x = Convolution2D(16, 3, 3, subsample=(2,2), border_mode="same", activation="relu")(x)
	x = Convolution2D(32, 3, 3, border_mode="same", activation="relu")(x)
	x = MaxPooling2D((3,3), strides=(2,2))(x)
	x = Convolution2D(64, 3, 3, subsample=(2,2), border_mode="same", activation="relu")(x)
	x = Convolution2D(64, 3, 3, subsample=(2,2), border_mode="same", activation="relu")(x)
	x = BatchNormalization()(x)
	x = factor_5x5(x,40,24,40,32,48)
	x = SpatialDropout2D(.1)(x)
	x = pool_5x5(x,64,48,64,64,96,32)
	x = SpatialDropout2D(.1)(x)
	x = factor_5x5(x,80,48,80,64,96)
	return x

def construct():
	imgs = Input(shape=(487, 866, 3))

	x = stem(imgs)

	# Detector. Approximates image's coverage matrix. We use weights from this layer 
	##			to restrict the classifier input to only features related to fish.
	conv_coverage = Convolution2D(1, 1, 1, activation='sigmoid')
	pred_mat = conv_coverage(x)
	pred_mat = Reshape((16,28),name="coverage")(pred_mat)

	# Classifier. Infers fish type.
	fishy_feats = Lambda(fishy_features, arguments={'tied_to': conv_coverage})(x)
	fishy_feats = Activation('relu')(fishy_feats)

	class_1 = pool_5x5(fishy_feats,64,48,64,64,96,32)
	class_1 = SpatialDropout2D(.2)(class_1)

	class_2 = pool_5x5(class_1,85,64,85,64,128,43)
	class_2 = SpatialDropout2D(.2)(class_2)

	class_3 = pool_5x5(class_2,106,80,106,106,160,54)
	class_3 = SpatialDropout2D(.2)(class_3)

	class_4 = pool_5x5(class_3,128,96,128,128,192,64)

	fcn = Flatten()(class_4)
	fcn = Dropout(.5)(fcn)
	fcn = Dense(128)(fcn)

	class_vec = Dense(8, activation='softmax', name="class")(fcn)

	return Model(input=imgs,output=[pred_mat,class_vec])

datagen_args = dict(rotation_range=65.,
					width_shift_range=0.1,
					height_shift_range=0.1,
					zoom_range=0.3,
					horizontal_flip=True,
					vertical_flip=True)

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 4:
		print "Usage: train nb_epoch batch_size samples_per_epoch"
		sys.exit()

	model = ClassifierContainer(name,construct(),32,"adam", datagen_args=datagen_args)
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
