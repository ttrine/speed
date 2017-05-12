import numpy as np
import imageio, cv2

def to_tensor(dataset):
	vid = imageio.get_reader('data/'+dataset+'.mp4', 'ffmpeg')

	data = np.zeros((len(vid),120,160,3),dtype=np.uint8)

	for i, img in enumerate(vid):
		data[i] = cv2.resize(img,(160,120))

	np.save('data/'+dataset,data)

to_tensor('train')
to_tensor('test')