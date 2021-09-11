import numpy as np
from PIL import Image
from glob import glob
from matplotlib import pyplot as plt

gt_dir='322_gt/*.png'

gt_imgs = glob(gt_dir)
gt_imgs.sort()

gts = []
idx = []

for i in range(len(gt_imgs)):
   gt = Image.open(gt_imgs[i]).convert('L')
   gt = gt.resize((160, 128))
   gt = np.array(gt)
   gt = gt.astype(np.float64)
   gt = np.reshape(gt, (np.shape(gt)[0]*np.shape(gt)[1], -1))
   gts.append(gt)

   ind = gt_imgs[i].split('.')[0][-4:]
   idx.append(int(ind))

np.stack(gts)
gts = np.reshape(gts, (np.shape(gts)[0], np.shape(gts)[1]))
gts = np.transpose(gts)

np.save(gt_dir.split('/')[0] + '_gt.npy', gts)
np.save(gt_dir.split('/')[0] + '_idx.npy', idx)