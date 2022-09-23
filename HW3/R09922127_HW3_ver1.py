import cv2
import numpy as np
from  matplotlib import pyplot as plt

# 讀取圖檔
img = cv2.imread('lena.bmp')
s = img.shape
# arr紀錄原圖hist
arr = np.zeros(256,dtype=int)
x_axes = np.arange(256)
for i in range(s[0]):
    for j in range(s[1]):
        arr[img[i,j,0]] += 1

# img2 is image with intensity divided by 3
img2 = img//3
plt.imshow(img2.astype('uint8'))

# arr2 紀錄 img2 的 hist
arr2 = np.zeros(256,dtype=int)
for i in range(s[0]):
    for j in range(s[1]):
        arr2[img2[i,j,0]] += 1

# 建立Histogram Equalization的字典
histequ = dict()
c = 255/(s[0]*s[1])
for i in range(256):
    histequ[i] = int(sum(arr2[0:i+1])*c)

# he 是做Histogram Equalization後的圖
he = np.zeros(img.shape)
for i in range(s[2]):
    for j in range(s[0]):
        for k in range(s[1]):
            he[j,k,i] = histequ[img2[j,k,i]]
# arr3 紀錄 he 的 hist
arr3 = np.zeros(256,dtype=int)
he1 = he.astype(int)
for i in range(s[0]):
    for j in range(s[1]):
        arr3[he1[i,j,0]] += 1

plt.subplot(3, 2, 1)
plt.title('origin')
plt.imshow(img.astype('uint8'))
plt.subplot(3, 2, 2)
plt.title('origin historgram')
plt.bar(x_axes,arr) #plt.hist
plt.xlabel('pixel')

plt.subplot(3, 2, 3)
plt.title('intensity divided by 3')
plt.imshow(img2.astype('uint8'))
plt.subplot(3, 2, 4)
plt.title('intensity divided by 3 historgram')
plt.bar(x_axes,arr2) #plt.hist
plt.xlabel('pixel')

plt.subplot(3, 2, 5)
plt.title('origin')
plt.imshow(he.astype('uint8'))
plt.subplot(3, 2, 6)
plt.title('origin historgram')
plt.bar(x_axes,arr3) #plt.hist
plt.xlabel('pixel')

plt.show()