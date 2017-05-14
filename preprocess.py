import numpy as np
import imageio, cv2

def normalize(train, test):
	''' Normalize feature-wise (RGB). '''

	mean = np.mean(train,(0,1,2))
	std = np.std(train,(0,1,2))

	train -= mean
	train /= std

	test -= mean
	test /= std

def to_tensor(dataset):
	vid = imageio.get_reader('data/'+dataset+'.mp4', 'ffmpeg')

	data = np.zeros((len(vid),120,160,3))

	for i, img in enumerate(vid):
		data[i] = cv2.resize(img,(160,120))

	return data

train = to_tensor('train')
test = to_tensor('test')

normalize(train,test)

np.save('data/train',train)
np.save('data/train',test)
