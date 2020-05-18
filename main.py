import matplotlib.pyplot as plt
from utilities import *

path = '' # current image path
im = plt.imread(path)
eq_im = ecualizaYUV(im)
plt.imshow(eq_im)
