from sklearn import preprocessing
import numpy as np
from sklearn.externals import joblib
import cv2
from PIL import Image
import time
import os
class SearchImage:

    def GetImageFeature(self,filename):
        print(os.path.abspath(os.curdir))
        words, path, idf, centers, k = joblib.load("bow.pkl")
        sift = cv2.xfeatures2d.SIFT_create()
        dst_img = Image.open(filename)
        dst_img = np.array(dst_img)
        kpt, dest = sift.detectAndCompute(dst_img, None)
        words_count=np.zeros(k)
        for d in dest:
            distance = np.power(np.tile(d, (centers.shape[0], 1)) - centers, 2).sum(axis=1)
            words_count[np.argmin(distance)] += 1
        words_count/=words_count.sum(axis=0)
        words_count*=idf
        words_count=np.expand_dims(words_count,axis=0)
        words_count=preprocessing.normalize(words_count,"l1")
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        words_count=words_count.astype('float32')
        words=words.astype('float32')
        matches = flann.knnMatch(words_count,words, k=8)
        count=1
        imgs=[]
        for i in matches[0]:
            count+=1
            imgs.append(path[i.trainIdx])
        #print(imgs)
        return imgs
'''
s=SearchImage()
t=s.GetImageFeature("/home/wlj/pic/test1/image.orig/0.jpg")
'''

