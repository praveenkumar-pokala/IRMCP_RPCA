import numpy as np
import cv2
import os
import shutil

input_vid = '322_gt.mp4'
path = input_vid.split('.')[0]

cap = cv2.VideoCapture(input_vid)

if os.path.isdir(path):
	shutil.rmtree(path)

os.mkdir(path)

ret = True
data = []

count = 0
im_path = path + "/gt_{:04d}.png"
while ret:
	ret, img = cap.read()
	if ret:
		count += 1
		img = cv2.resize(img, (160, 128))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		cv2.imwrite(im_path.format(count), img)