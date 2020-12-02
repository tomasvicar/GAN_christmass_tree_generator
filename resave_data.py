import glob
import os
from skimage.io import imsave,imread
import matplotlib.pyplot as plt
import cv2
import numpy as np

data_path = '../data_orig'
save_path = '../data_64'


img_size = 64

try:
    os.mkdir(save_path)
except:
    pass



file_names = glob.glob(data_path + '/**/*',recursive=True)


num = -1
for file_num, file_name in enumerate(file_names):
    
    try:
    
        img = imread(file_name)
        
        img = img[:,:,:3]
        
        insize = img.shape[0:2]
        
        
        out = np.zeros((np.max(insize),np.max(insize),3),dtype=np.uint8)
        
        out[:insize[0],:insize[1],:] = img
        
        
        out = cv2.resize(out,(img_size, img_size),interpolation=cv2.INTER_LINEAR)
        
        num = num+1
        
        imsave(save_path + '/img' + str(num).zfill(6)  + '.png',out)
        
        
    except:
        
        pass
    
    