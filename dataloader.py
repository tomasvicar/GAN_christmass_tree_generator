import glob
import os
from skimage.io import imread
import torch
import numpy as np

class DataLoader(torch.utils.data.Dataset):
    def __init__(self,path):
        self.path=path

        self.files_img=glob.glob(self.path + '/*.png')*100

        self.num_of_imgs=len(self.files_img)

    def __len__(self):
        return self.num_of_imgs


    def __getitem__(self, index):
        
        img = imread(self.files_img[index])
        
        img = np.transpose(img,(2, 0, 1))

        img=torch.Tensor((img.astype(np.float32)/255-0.5)/0.5)
        
    
        return img