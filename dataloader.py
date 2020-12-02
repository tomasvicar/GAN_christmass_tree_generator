import glob
import os
from skimage.io import imread, imsave
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

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
    
    
    
    
    
    
if __name__ == "__main__": 
    loader = DataLoader('../data_64')
    loader = torch.utils.data.DataLoader(loader, batch_size=128,shuffle=True, num_workers=0,drop_last=True)
    
    for i,data in enumerate(loader):
        
        plt.figure(figsize=(15,15))
        img = np.transpose(vutils.make_grid(data.cpu().detach()[:32],padding=2, normalize=True).numpy(),(1,2,0))
        plt.imshow(img)
        plt.show()
        imsave('example_img/real' + str(i).zfill(7) + '.png',img)
    
        if i ==3: 
            break
        
    