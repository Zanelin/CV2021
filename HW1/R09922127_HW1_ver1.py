import cv2
import numpy as np
from  matplotlib import pyplot as plt

#read original file
img = cv2.imread('lena.bmp')

ud = np.zeros(img.shape) #upside-down
rl = np.zeros(img.shape) #right-side-left
df = np.zeros(img.shape) #iagonally flip

for i in range(s[2]):
    for j in range(s[0]):
        for k in range(s[1]):
            ud[j,k,i] = img[s[0]-j-1,k,i]
            rl[j,k,i] = img[j,s[1]-k-1,i]
            df[j,k,i] = img[s[0]-j-1,s[1]-k-1,i]
#save
cv2.imwrite('upside-down_lena.bmp', ud)
cv2.imwrite('right-side-left_lena.bmp', rl)
cv2.imwrite('diagonally_flip_lena.bmp', df)

#----------------------------------------------
#rotate lena.bmp 45 degrees clockwise
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w//2, h//2), -45, 1)
rot = cv2.warpAffine(img, M, (w, h))

#shrink lena.bmp in half
shr = np.zeros((img.shape[0]//2,img.shape[1]//2,img.shape[2]))
for i in range(s[2]):
    for j in range(s[0]//2):
        for k in range(s[1]//2):
            shr[j,k,i] = img[j*2,k*2,i]

#binarize lena.bmp at 128 to get a binary image
bn = np.zeros(img.shape)
for i in range(s[2]):
    for j in range(s[0]):
        for k in range(s[1]):
            if img[j,k,i]>128:
                bn[j,k,i] = 255
            else:
                bn[j,k,i] = 0

#save
cv2.imwrite('rotate_lena.bmp', rot)
cv2.imwrite('shrink_lena.bmp', shr)
cv2.imwrite('binarize_lena.bmp', bn)
