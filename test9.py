import numpy as np
import cv2
from PIL import Image
import time
import copy
import imageio

class seamcarving:

    def __init__(self):
        self.iteration=200

    def get_sobel(self,img):

        img=img.astype('uint8')

        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)

        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        return dst.astype('int64')

    def DynamicProgram(self,dst,img,originimage):
        rows, cols = dst.shape
        dst = np.pad(dst, ((1, 0), (1, 1)),'constant',constant_values=999999999)
        dst_min = dst.copy()
        for i in range(2,rows+1):
            for j in range(1,cols+1):
                dst_min[i][j]=dst[i][j]+np.min(dst_min[i-1,j-1:j+2])
        mask=(dst==dst)
        j=np.argmin(dst_min[rows])
        mask[rows][j]=False
        for i in range(rows-1,0,-1):
            j+=np.argmin(dst_min[i,j-1:j+2])-1
            mask[i][j]=False
        mask=mask[1:rows+1,1:cols+1]
        img=img[mask].reshape(rows,cols-1)
        originimage=originimage[mask].reshape(rows,cols-1,3)
        return img,originimage

    def processing(self):
        oriimg=Image.open("/home/wlj/pic/5.jpg")
        img=oriimg.convert('L')
        oriimg=np.array(oriimg)
        img=np.array(img)
        for i in range(self.iteration):
            dst=self.get_sobel(img)
            img,oriimg=self.DynamicProgram(dst,img,oriimg)

        Image.fromarray(oriimg).show()
tic=time.time()
s=seamcarving()
im=np.array([[5,8,12,3],[4,2,3,9],[7,3,4,2],[4,5,7,8]])
s.processing()
toc=time.time()
print(toc-tic)
