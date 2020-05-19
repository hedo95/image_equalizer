import matplotlib.pyplot as plt
from utilities import *

path = '' # current image path
format = '' # image format to be saved
im = plt.imread(path)
eq_im = ecualizaYUV(im)
plt.imsave(path,eq_im,format=format)
plt.imshow(eq_im)
