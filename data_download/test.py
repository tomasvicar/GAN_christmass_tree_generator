import glob 

from skimage.io import imread


import matplotlib.pyplot as plt



names = glob.glob('christmas tree/*')


for name in names:
    
    data = imread(name)
    plt.imshow(data)
    plt.show()


