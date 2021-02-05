import os
from scipy.io import loadmat as sio
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

path = 'D:/Downloads/trainval/trainval'
files = os.listdir(path)
labels_path = os.path.join(path, 'labels')

for afile in files:
	file_path = os.path.join(path, afile)
	if os.path.isfile(file_path):
		if os.path.getsize(file_path) == 0:
			continue
		mat_idx = afile[:afile.find('.mat')]
		mat_file = sio(file_path)
		mat_file = np.array(mat_file['LabelMap'])
		mat_file = mat_file.astype(np.uint16)
		dst_path = os.path.join(labels_path, mat_idx + '.npy')
		np.save(dst_path, mat_file)
