import cv2
import numpy as np
from  matplotlib import pyplot as plt

# Åª¨ú¹ÏÀÉ
img = cv2.imread('lena.bmp')

def GetGaussianNoice_Image(original_image, amp = 10):
    result = original_image.copy()
    for r in range(original_image.shape[0]):
        for c in range(original_image.shape[1]):
            noisePixel = int(original_image[r,c,0] + amp * np.random.normal(0,1))
            if noisePixel > 255:
                noisePixel = 255
            result[r,c,:] = noisePixel
    return result

def SNR(ori_img, noi_img):
    tmp1 = ori_img[:,:,0]/255
    tmp2 = noi_img[:,:,0]/255
    mu = np.sum(tmp1)/(512*512)
    vs = np.sum((tmp1-mu)**2)/(512*512)
    mu_noi = np.sum(tmp2 - tmp1)/(512*512)
    vn = np.sum((tmp2-tmp1-mu_noi)**2)/(512*512)
    return 20*np.log10(np.sqrt(vs/vn))

def box_filter(ori_img, s):
    div = s * s
    bf_result = ori_img.copy()
    h = s//2
    for r in range(h,ori_img.shape[0] - h):
        for c in range(h,ori_img.shape[1] - h):
            #print(ori_img[r-h:r+h+1, c-h:c+h+1, 0].shape)
            bf_result[r, c, :] = np.sum(ori_img[r-h:r+h+1, c-h:c+h+1, 0])/div 
    return bf_result

def mid_filter(ori_img, s):
    div = s * s
    mf_result = ori_img.copy()
    h = s//2
    for r in range(h,ori_img.shape[0] - h):
        for c in range(h,ori_img.shape[1] - h):
            mf_result[r, c, :] = np.sort(ori_img[r-h:r+h+1, c-h:c+h+1, 0].reshape(div))[div//2]
    return mf_result

def dilation(img,k): 
    img_pad = np.pad(array=img, pad_width=((3,3),(3,3),(0,0)), mode='constant', constant_values=0)
    img_tmp = np.zeros(img_pad.shape)
    for i in range(3,515): #3~514
        for j in range(3,515): #3~514
            a1 = max(img_pad[i-2,j-1:j+3,0])
            a2 = max(img_pad[i-1,j-2:j+3,0])
            a3 = max(img_pad[i,j-2:j+3,0])
            a4 = max(img_pad[i+1,j-2:j+3,0])
            a5 = max(img_pad[i+2,j-1:j+3,0])
            f  = max(img_pad[i,j:,0])
            f = 0
            img_tmp[i,j,0] = max(a1,a2,a3,a4,a5,f)
            img_tmp[i,j,1] = max(a1,a2,a3,a4,a5,f)
            img_tmp[i,j,2] = max(a1,a2,a3,a4,a5,f)
    return img_tmp[3:515,3:515]
def erosion(img,k): 
    img_pad = np.pad(array=img, pad_width=((3,3),(3,3),(0,0)), mode='constant', constant_values=255)
    img_tmp = np.zeros(img_pad.shape)
    for i in range(3,515): #3~514
        for j in range(3,515): #3~514
            a1 = min(img_pad[i-2,j-1:j+3,0])
            a2 = min(img_pad[i-1,j-2:j+3,0])
            a3 = min(img_pad[i,j-2:j+3,0])
            a4 = min(img_pad[i+1,j-2:j+3,0])
            a5 = min(img_pad[i+2,j-1:j+3,0])
            f  = min(img_pad[i,j:,0])
            f = 255
            img_tmp[i,j,0] = min(a1,a2,a3,a4,a5,f)
            img_tmp[i,j,1] = min(a1,a2,a3,a4,a5,f)
            img_tmp[i,j,2] = min(a1,a2,a3,a4,a5,f)
    return img_tmp[3:515,3:515]
def opening(img,k):
    return dilation(erosion(img,k),k)
def closing(img,k):
    return erosion(dilation(img,k),k)

kernel = np.array([[0,255,255,255,0],[255,255,255,255,255],[255,255,255,255,255],[255,255,255,255,255],[0,255,255,255,0]])

img_Gau10 = GetGaussianNoice_Image(img, 10)
print(SNR(img, img_Gau10))
plt.imshow(img_Gau10.astype('uint8'))
plt.show()

#img_Gau10, img_Gau30, img_SAP005, img_SAP01
bf13 = box_filter(img_Gau10, 3)
print(SNR(img, bf13))
plt.imshow(bf13.astype('uint8'))
plt.show()

bf15 = box_filter(img_Gau10, 5)
print(SNR(img, bf15))
plt.imshow(bf15.astype('uint8'))
plt.show()

mf13 = mid_filter(img_Gau10, 3)
print(SNR(img, mf13))
plt.imshow(mf13.astype('uint8'))
plt.show()

mf15 = mid_filter(img_Gau10, 5)
print(SNR(img, mf15))
plt.imshow(mf15.astype('uint8'))
plt.show()

img_oc = closing(opening(img_Gau10,kernel),kernel)
print(SNR(img, img_oc))
plt.imshow(img_oc.astype('uint8'))
plt.show()

img_co = opening(closing(img_Gau10,kernel),kernel)
print(SNR(img, img_co))
plt.imshow(img_co.astype('uint8'))
plt.show()

###################################

img_Gau30 = GetGaussianNoice_Image(img, 30)
print(SNR(img, img_Gau30))
plt.imshow(img_Gau30.astype('uint8'))
plt.show()

bf23 = box_filter(img_Gau30, 3)
print(SNR(img, bf23))
plt.imshow(bf23.astype('uint8'))
plt.show()

bf25 = box_filter(img_Gau30, 5)
print(SNR(img, bf25))
plt.imshow(bf25.astype('uint8'))
plt.show()

mf23 = mid_filter(img_Gau30, 3)
print(SNR(img, mf23))
plt.imshow(mf23.astype('uint8'))
plt.show()

mf25 = mid_filter(img_Gau30, 5)
print(SNR(img, mf25))
plt.imshow(mf25.astype('uint8'))
plt.show()

img_oc = closing(opening(img_Gau30,kernel),kernel)
print(SNR(img, img_oc))
plt.imshow(img_oc.astype('uint8'))
plt.show()

img_co = opening(closing(img_Gau30,kernel),kernel)
print(SNR(img, img_co))
plt.imshow(img_co.astype('uint8'))
plt.show()

##################################

img_SAP005 = GetSaltAndPepper_Image(img, 0.05)
print(SNR(img, img_SAP005))
plt.imshow(img_SAP005.astype('uint8'))
plt.show()

bf33 = box_filter(img_SAP005, 3)
print(SNR(img, bf33))
plt.imshow(bf33.astype('uint8'))
plt.show()

bf35 = box_filter(img_SAP005, 5)
print(SNR(img, bf35))
plt.imshow(bf35.astype('uint8'))
plt.show()

mf33 = mid_filter(img_SAP005, 3)
print(SNR(img, mf33))
plt.imshow(mf33.astype('uint8'))
plt.show()

mf35 = mid_filter(img_SAP005, 5)
print(SNR(img, mf35))
plt.imshow(mf35.astype('uint8'))
plt.show()

img_oc = closing(opening(img_SAP005,kernel),kernel)
print(SNR(img, img_oc))
plt.imshow(img_oc.astype('uint8'))
plt.show()

img_co = opening(closing(img_SAP005,kernel),kernel)
print(SNR(img, img_co))
plt.imshow(img_co.astype('uint8'))
plt.show()

#################################

img_SAP01 = GetSaltAndPepper_Image(img)
print(SNR(img, img_SAP01))
plt.imshow(imgSAP.astype('uint8'))
plt.show()

bf43 = box_filter(img_SAP01, 3)
print(SNR(img, bf43))
plt.imshow(bf43.astype('uint8'))
plt.show()

bf45 = box_filter(img_SAP01, 5)
print(SNR(img, bf45))
plt.imshow(bf45.astype('uint8'))
plt.show()

mf43 = mid_filter(img_SAP01, 3)
print(SNR(img, mf43))
plt.imshow(mf43.astype('uint8'))
plt.show()

mf45 = mid_filter(img_SAP01, 5)
print(SNR(img, mf45))
plt.imshow(mf45.astype('uint8'))
plt.show()

img_oc = closing(opening(img_SAP01,kernel),kernel)
print(SNR(img, img_oc))
plt.imshow(img_oc.astype('uint8'))
plt.show()

img_co = opening(closing(img_SAP01,kernel),kernel)
print(SNR(img, img_co))
plt.imshow(img_co.astype('uint8'))
plt.show()






