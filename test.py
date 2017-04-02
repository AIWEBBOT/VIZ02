import numpy as np 
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('train_cv_img.jpg')
mask = np.zeros(img.shape[:2] , np.uint8) 

bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

rect = (300 , 120 , 470 , 350)

#this modifies mask
cv2.grabCut(img,mask,rect,bgdModel, fgdModel , 5 , cv2.GC_INIT_WITH_RECT)

#If mask==2 or mask==1 , mask2 get 0, otherwise it gets 1 as 'uint8' type
mask2 = np.where((mask==2) | (mask==0),0,1).astype('uint8')

#adding additional dimension for rgb to the mask, by default it gets 1
#multiply with input image to get the segmented image
img_cut = img*mask2[: , : , np.newaxis]

plt.subplot(211),plt.imshow(img)
plt.title('Input Image') , plt.xticks([]),plt.yticks([])
plt.subplot(212),plt.imshow(img_cut)
plt.title('Grab cut'), plt.xticks([]),plt.yticks([])
plt.show()
