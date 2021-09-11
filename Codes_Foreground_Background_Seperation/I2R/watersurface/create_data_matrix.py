import os
import os.path
import numpy as np
from PIL import Image

root_dir='WaterSurface'
data=[]
retour = []
for (root,dirs,files) in os.walk(root_dir):
    for f in files:
        retour.append(f)

retour.sort()
idx = np.zeros(len(retour))
for i in range(len(retour)):        
   img = Image.open(root + '/' + retour[i]).convert('L')
   img = np.array(img)
   img = img.astype(np.float64)
   img = np.reshape(img, (np.shape(img)[0]*np.shape(img)[1], -1))
   data.append(img)
   img_name = retour[i].split('.')[0]
   idx[i] = int(img_name[-4:])

np.stack(data)
data = np.reshape(data, (np.shape(data)[0], np.shape(data)[1]))
data = np.transpose(data)
np.save(root_dir + '.npy', data)
np.save('idx.npy', idx)
