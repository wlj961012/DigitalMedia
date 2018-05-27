from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
class colortrast:
    def histeq(self,img,nbr_bins=256):
        im=img.copy()
        times = np.bincount(im.flatten(), minlength=255)  # 计算各灰度级出现次数
        freq = times / (im.size)  # 计算出现频率
        accumu = freq.cumsum(axis=0)  # 累加频率
        for idx, pixel in enumerate(im):
            im[idx] = np.uint8(accumu[pixel] * 255 + 0.5)
        return im.reshape(im.shape).astype('uint8')
    def histeq1(self,img):
        im=img.copy()
        H,W,_=im.shape
        r=self.histeq(im[:,:,0]).flatten()
        g = self.histeq(im[:, :, 1]).flatten()
        b = self.histeq(im[:, :, 2]).flatten()
        im=np.column_stack((r,g,b))
        im.resize(H,W,3)
        return im.astype('uint8')
    def linear_contrast_stretch(self,fa,ga,fb,gb,im):
        im2=im.copy()
        H,W,_=im2.shape
        k1=ga/fa
        k2=(gb-ga)/(fb-fa)
        k3=(255-gb)/(255-fb)
        for row in range(H):
            for col in range(W):
                for i in range(3):
                    if im2[row][col][i] < fa:
                        im2[row][col][i] = im2[row][col][i] * k1
                    elif im2[row][col][i] > fb:
                        im2[row][col][i] = (im2[row][col][i] - fb) * k3 + gb
                    else:
                        im2[row][col][i] = (im2[row][col][i] - fa) * k2 + ga
        return im2.astype('uint8')
img=Image.open('/home/wlj/pic/6.jpg')
img=np.array(img)
plt.figure()
plt.subplot(421)
plt.title('origin')
plt.axis('off')
plt.imshow(img)
plt.subplot(422)
arr=img.flatten()
plt.hist(arr, bins=256, normed=1, facecolor='red', alpha=0.75)
plt.subplot(423)
plt.title('linear')
plt.axis('off')
t=colortrast()
im2=t.linear_contrast_stretch(70,20,180,230,img)
plt.imshow(im2)
plt.subplot(424)
arr2=im2.flatten()
plt.hist(arr2, bins=256, normed=1, facecolor='red', alpha=0.75)
plt.subplot(425)
plt.axis('off')
plt.title('histrgb')
im3=t.histeq1(img)
plt.imshow(im3)
plt.subplot(426)
arr3=im3.flatten()
plt.hist(arr3, bins=256, normed=1, facecolor='red', alpha=0.75)
plt.subplot(427)
plt.axis('off')
plt.title('hist')
im4=t.histeq(img)
plt.imshow(im4)
plt.subplot(428)
arr4=im4.flatten()
plt.hist(arr4, bins=256, normed=1, facecolor='red', alpha=0.75)

plt.show()
