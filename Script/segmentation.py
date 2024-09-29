from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from skimage.util import invert
import codecs
import cv2

#with codecs.open('image_' + str(1)  + '.txt', encoding='utf-8-sig') as f:
#    im = np.loadtxt(f)

#skeleton = skeletonize(im)
#fig1 = plt.figure() 
#plt.matshow(skeleton, cmap=matplotlib.cm.Greys_r, fignum = 0)
#plt.show()

mask = cv2.imread(".\Data\mask_1.tiff")
mask = np.sum(mask, axis = 2)
mask[mask > 0] = 1
mask