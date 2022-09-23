import cv2
import numpy as np
from  matplotlib import pyplot as plt

def dilation(img,k): 
    img_pad = np.pad(array=img, pad_width=((3,3),(3,3),(0,0)), mode='constant', constant_values=255)
    img_tmp = np.zeros(img_pad.shape)
    for i in range(3,515): #3~514
        for j in range(3,515): #3~514
            if img_pad[i,j,0] != 0:
                for p in range(5):
                    for q in range(5):
                        if k[p,q] != 0:
                            img_tmp[i-2+p,j-2+q,0] = 255
                            img_tmp[i-2+p,j-2+q,1] = 255
                            img_tmp[i-2+p,j-2+q,2] = 255
    return img_tmp[3:515,3:515]
                            
def erotion(img,k): 
    img_pad = np.pad(array=img, pad_width=((3,3),(3,3),(0,0)), mode='constant', constant_values=255)
    img_tmp = np.zeros(img_pad.shape)
    num = np.count_nonzero(k == 255)
    for i in range(3,515): #3~514
        for j in range(3,515): #3~514
            c = 0
            for p in range(5):
                for q in range(5):
                    if k[p,q] != 0 and k[p,q] == img_pad[i-2+p,j-2+q,0]:
                        c+=1
            if c == num:
                img_tmp[i,j,0] = 255
                img_tmp[i,j,1] = 255
                img_tmp[i,j,2] = 255
    return img_tmp[3:515,3:515]

def opening(img,k):
    return dilation(erotion(img,k),k)

def closing(img,k):
    return erotion(dilation(img,k),k)

def hit_and_miss_trans(img,k):
    img_c =  -(img-255)
    J = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,255,255,0,0],[0,0,255,0,0],[0,0,0,0,0]])
    K = np.array([[0,0,0,0,0],[0,0,255,255,0],[0,0,0,255,0],[0,0,0,0,0],[0,0,0,0,0]])
    a_j = erotion(img,J)
    
    ac_k = erotion(img_c,K)
    
    return (a_j*ac_k/255).astype(int)

# 讀取圖檔
img = cv2.imread('lena.bmp')

s = img.shape

bn = np.zeros(img.shape)
bnc = np.zeros(img.shape)
for i in range(s[2]):
    for j in range(s[0]):
        for k in range(s[1]):
            if img[j,k,i]>127:
                bn[j,k,i] = 255
                bnc[j,k,i] = 0
            else:
                bn[j,k,i] = 0
                bnc[j,k,i] = 255

kernel = np.array([[0,255,255,255,0],[255,255,255,255,255],[255,255,255,255,255],[255,255,255,255,255],[0,255,255,255,0]])


img_hnmt = hit_and_miss_trans(bn,kernel)
plt.imshow(img_hnmt.astype('uint8'))
plt.show()