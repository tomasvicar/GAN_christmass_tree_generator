import glob
import os
from skimage.io import imread
import numpy as np

path = '/content/drive/MyDrive/tmp_imgs'

names = glob.glob(path + '/*.png')



# import imageio
# with imageio.get_writer('/content/drive/MyDrive/gan_res.gif', mode='I') as writer:
#     for i,filename in enumerate(names):
#         print(i)
#         image = imageio.imread(filename)
#         writer.append_data(image)




import imageio

images = []
for i,file in enumerate(names[0:50]+names[50:150:2] +names[150:300:3] +names[300:-1:5]):
    print(i)
    images.append(imageio.imread(file))


imageio.mimwrite('gan_res2.gif', images, fps=3)
