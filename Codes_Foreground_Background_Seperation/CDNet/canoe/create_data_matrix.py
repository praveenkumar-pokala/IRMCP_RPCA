import numpy as np
from PIL import Image
from glob import glob
from matplotlib import pyplot as plt

input_dir='canoe/input/*.jpg'
gt_dir='canoe/groundtruth/*.png'

input_imgs = glob(input_dir)
gt_imgs = glob(gt_dir)
input_imgs.sort()
gt_imgs.sort()

data = []
gts = []
idx = []

for i in range(len(input_imgs)):        
   img = Image.open(input_imgs[i]).convert('L')
   img = img.resize((160, 128))
   img = np.array(img)
   img = img.astype(np.float64)
   img = np.reshape(img, (np.shape(img)[0]*np.shape(img)[1], -1))
   data.append(img)

for i in range(len(gt_imgs)):
   gt = Image.open(gt_imgs[i]).convert('L')
   gt = gt.resize((160, 128))
   gt = np.array(gt)
   gt = gt.astype(np.float64)
   gt = np.reshape(gt, (np.shape(gt)[0]*np.shape(gt)[1], -1))
   gts.append(gt)

   ind = gt_imgs[i].split('.')[0][-5:]
   idx.append(int(ind))

np.stack(data)
np.stack(gts)
data = np.reshape(data, (np.shape(data)[0], np.shape(data)[1]))
data = np.transpose(data)
gts = np.reshape(gts, (np.shape(gts)[0], np.shape(gts)[1]))
gts = np.transpose(gts)

np.save(input_dir.split('/')[1] + '.npy', data)
np.save(input_dir.split('/')[1] + '_gt.npy', gts)
np.save(input_dir.split('/')[1] + '_idx.npy', idx)