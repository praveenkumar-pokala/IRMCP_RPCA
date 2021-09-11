import numpy as np
import cv2

input_dir = '322.mp4'
cap = cv2.VideoCapture(input_dir)

ret = True
data = []

while ret:
	ret, img = cap.read()
	if ret:
		img = cv2.resize(img, (160, 128))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = np.array(img)
		img = img.astype(np.float64)
		img = np.reshape(img, (np.shape(img)[0]*np.shape(img)[1], -1))
		data.append(img)

np.stack(data)
data = np.reshape(data, (np.shape(data)[0], np.shape(data)[1]))
data = np.transpose(data)
print(np.shape(data))
np.save(input_dir.split('.')[0] + '.npy', data)