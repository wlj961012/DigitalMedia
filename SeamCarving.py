import numpy as np
import cv2
from PIL import Image
import time
import copy
import imageio

class SeamCarving:

    def __init__(self):
        self.iteration=100
        self.imgs=[]

    def GetGradian(self,img):

        img=cv2.boxFilter(img,-1,(5,5))
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

        absX = cv2.convertScaleAbs(x)  # 转回uint8
        absY = cv2.convertScaleAbs(y)

        dst = cv2.addWeighted(absX,0.5, absY,0.5, 0).astype('int64')

        return dst

    def DynamicProgram(self,dst,img,originimage):
        #for i in range(3):
        rows,cols=img.shape
        dst = np.pad(dst, ((1, 0), (1, 1)),'constant',constant_values=999999999)
        dst_min = dst.copy()
        for i in range(2,rows+1):
            for j in range(1,cols+1):
                dst_min[i][j]=dst[i][j]+min(dst_min[i-1,j-1:j+2])
        mask=(dst==dst)
        j=np.argmin(dst_min[rows])
        mask[rows][j]=False
        for i in range(rows-1,0,-1):
            j+=np.argmin(dst_min[i,j-1:j+2])-1
            mask[i][j]=False
        mask=mask[1:rows+1,1:cols+1]
        img=img[mask].reshape(rows,cols-1)
        originimage=originimage[mask].reshape(rows,cols-1,3)
        self.iteration+=1
        if self.iteration<200:
            return self.GetGradian(img,originimage)
        else:
            Image.fromarray(originimage).show()
            return None

    def carving(self,dst,img,imgg):
        #for i in range(5):
            #self.iteration+=1
        dst_min=dst.copy()
        rows,cols=dst.shape
        for i in range(1,rows):
            for j in range(cols):
                m=min(dst_min[i-1,max(j-1,0):min(j+1,cols-1)+1])
                dst_min[i,j]=dst[i,j]+m
        mask=(dst_min==dst_min)
        minnum=np.min(dst_min[rows-1,:])
        j=0
        for j in range(cols):
            if dst_min[rows-1,j]==minnum:
                mask[rows-1,j]=False
                break
        for i in range(rows-1,0,-1):
            minnum=np.min(dst_min[i-1,max(0,j-1):min(cols-1,j+1)+1])
            l=len(dst_min[i-1,max(0,j-1):min(cols-1,j+1)+1])
            for k in range(l):
                if minnum==dst_min[i-1,max(0,j-1)+k]:
                    mask[i - 1, max(0, j - 1)+k]=False
                    j=max(0,j-1)+k
                    break
                '''if self.iteration>0:
                    imgg2=copy.deepcopy(imgg)
                    imgg2[~mask]=255
                    Image.fromarray(imgg2).save("tmp_trace.jpg",format='JPEG')
                    img_trace=imageio.imread("tmp_trace.jpg",format='JPEG')
                    self.imgs.append(img_trace)
                '''
        print(mask.shape,imgg.shape)
        imgg=imgg[mask].reshape(rows,cols-1,3)
        img=img[mask].reshape(rows,cols-1)
        dst=dst[mask].reshape(rows,cols-1)

        return img,imgg

        '''if self.iteration<100:
            print(self.iteration)
            return self.GetGradian(img,imgg)
        else:
            #imageio.mimsave("2.gif",self.imgs, 'GIF', duration=0.1)
            Image._show(Image.fromarray(imgg))
            return t,imgg'''

    def processing(self):
        oriimg=Image.open("/home/wlj/pic/5.jpg")
        img=oriimg.convert('L')
        oriimg=np.array(oriimg)
        img=np.array(img)
        for i in range(self.iteration):
            dst=self.GetGradian(img)
            img,oriimg=self.carving(dst,img,oriimg)
        img=img.T
        oriimg=oriimg.transpose(1,0,2)
        for i in range(self.iteration):
            dst=self.GetGradian(img)
            img,oriimg=self.carving(dst,img,oriimg)
        Image.fromarray(oriimg.transpose(1,0,2)).show()

tic=time.time()
s=SeamCarving()
s.processing()
toc=time.time()
print(toc-tic)

'''
tic=time.time()
imgg=Image.open("/home/wlj/pic/5.jpg")
img=imgg.convert('L')
imgg=np.array(imgg)
img=np.array(img)
t=SeamCarving()
img=cv2.bilateralFilter(img,5,10,30)
t.GetGradian(img)
toc=time.time()
print(toc-tic)
'''