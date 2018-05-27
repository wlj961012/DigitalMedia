from PIL import Image
import numpy as np
import math
class ColorContrast:
    def __init__(self,imgpath):
        self._imgpath=imgpath
        img = Image.open(self._imgpath).convert('RGB')
        self.npimg=np.array(img)

    def change_contrast(self,delta):
        if len(self.npimg.shape)==2:
            return self._grey_contrast(delta)
        else:
            return self._rgb_contrast(delta)
    def _grey_contrast(self,delta):
        mean=np.array([self.npimg[:,:].mean()])
        ret_img=self.npimg.astype(np.float32)-mean
        ret_img*=delta
        ret_img+=mean
        ret_img[ret_img > 255] = 255
        ret_img[ret_img < 0] = 0
        return Image.fromarray(ret_img.astype('uint8'))#将数组转换成图片格式
    def _rgb_contrast(self,ratio):
        mean=np.array([self.npimg[:,:,0].mean(),self.npimg[:,:,1].mean(),self.npimg[:,:,2].mean()])
        ret_img=self.npimg.astype(np.float32)-mean
        ret_img*=ratio
        ret_img+=mean
        ret_img[ret_img>255]=255
        ret_img[ret_img<0]=0
        return Image.fromarray(ret_img.astype(np.uint8))
    #调节饱和度和亮度
    def RGB_to_HSL_optimize(self,img):  # 将RGB图像转换成HSL图像
        H, W, _ = img.shape
        img = img.astype(float)
        img /= 255
        im = img.copy()#reserve img value for val_R ,val_B,val_G
        var_min = np.min(img, axis=2)#get RGB min
        var_max = np.max(img, axis=2)#get RGB max
        del_max = var_max - var_min
        img[:, :, 2] = (var_max + var_min) / 2.0
        img[:, :, 0] = 0.0
        img[:, :, 1] = 0.0
        mask1 = (del_max != 0.0)
        mask2 = (img[:, :, 2] < 0.5)
        s = img[:, :, 1]
        l = img[:, :, 2]
        s[mask1 & mask2] = (del_max / (var_max + var_min))[mask1 & mask2]
        s[mask1 & (~mask2)] = (del_max / (2.0 - var_max - var_min))[mask1 & (~mask2)]
        del_R = (((var_max - im[:, :, 0]) / 6.0) + (del_max / 2.0)) / del_max
        del_G = (((var_max - im[:, :, 1]) / 6.0) + (del_max / 2.0)) / del_max
        del_B = (((var_max - im[:, :, 2]) / 6.0) + (del_max / 2.0)) / del_max
        h = img[:, :, 0]
        mask3 = (im[:, :, 0] == var_max)
        h[mask3 & mask1] = (del_B - del_G)[mask3 & mask1]
        mask4 = (im[:, :, 1] == var_max)
        h[mask4 & mask1] = ((1.0 / 3.0) + del_R - del_B)[mask4 & mask1]
        mask5 = (im[:, :, 2] == var_max)
        h[mask5 & mask1] = ((2.0 / 3.0) + del_G - del_R)[mask5 & mask1]
        h[np.where(h < 0.0)] = np.modf(h)[0][np.where(h < 0.0)]+1
        h[np.where(h > 1.0)] = np.modf(h)[0][np.where(h > 1.0)]-1
        # print(np.stack[:2,:2,:])
        return np.dstack((h, s, l))
    def HSL_to_RGB_optimize(self,img):
        def Hue_2_RGB_optimize(v1, v2, vh):
            mask_vh0 = [vh < 0.0]
            mask_vh1 = [vh > 1.0]
            vh[mask_vh0] = np.modf(vh)[0][mask_vh0] + 1
            vh[mask_vh1] = np.modf(vh)[0][mask_vh1]
            ret =v1.copy()
            mask1 = 6 * vh < 1.0
            mask2 = (2 * vh < 1.0) & (~mask1)
            mask3 = (3 * vh < 2.0) & (~mask1) & (~mask2)
            ret[mask1] = (v1 + (v2 - v1) * 6.0 * vh)[mask1]
            ret[mask2] = v2[mask2]
            ret[mask3] = (v1 + (v2 - v1) * ((2.0 / 3.0) - vh) * 6.0)[mask3]
            return ret

        h, s, l = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        r = l * 255
        g = l * 255
        b = l * 255
        var_2 = np.zeros(r.shape)
        var_1 = np.zeros(r.shape)
        mask1 = (s != 0.0)
        mask2 = (l < 0.5)
        var_2[mask1 & mask2] = (l * (1.0 + s))[mask1 & mask2]
        var_2[mask1 & (~mask2)] = ((l + s) - (s * l))[mask1 & (~mask2)]
        var_1[mask1] = (2.0 * l - var_2)[mask1]

        r[mask1] = (255 * Hue_2_RGB_optimize(var_1, var_2, h + (1.0 / 3.0)))[mask1]
        g[mask1] = (255 * Hue_2_RGB_optimize(var_1, var_2, h))[mask1]
        b[mask1] = (255 * Hue_2_RGB_optimize(var_1, var_2, h - (1.0 / 3.0)))[mask1]

        img[:, :, 0], img[:, :, 1], img[:, :, 2] = r, g, b
        #print(img[:2, :2, :])
        return img
    def change_bright(self,bri):
        img=self.npimg
        img=self.RGB_to_HSL_optimize(img)
        img[:,:,2]+=bri
        img=self.HSL_to_RGB_optimize(img)
        img[img > 255] = 255
        img[img < 0] = 0
        #print(img)
        return Image.fromarray(img.astype('uint8'))
    def change_saturation(self,sat):
        img = self.npimg
        img = self.RGB_to_HSL_optimize(img)
        img[:, :, 1] += sat
        img[img>1]=1
        img[img<0]=0
        img = self.HSL_to_RGB_optimize(img)
        img[img > 255] = 255
        img[img < 0] = 0
        #print(img)
        return Image.fromarray(img.astype('uint8'))
    #直方图均衡化
    def histeq(self,nbr_bins=256):
        im=self.npimg.copy()
        times = np.bincount(im.flatten(), minlength=255)  # 计算各灰度级出现次数
        freq = times / (im.size)  # 计算出现频率
        accumu = freq.cumsum(axis=0)  # 累加频率
        for idx, pixel in enumerate(im):
            im[idx] = np.uint8(accumu[pixel] * 255 + 0.5)
        return Image.fromarray(im.reshape(im.shape).astype('uint8'))
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
    #滤波算法
    def average_filter(self):# 均值滤波（i,j) 等于周围8个像素的平均值
        rows, cols, channels = self.npimg.shape
        img2 =self.npimg.copy()
        img2 = img2.astype('int64')
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                img2[i, j, :] = img2[i - 1:i + 2, j - 1:j + 2].mean(axis=0).mean(axis=0)
        return Image.fromarray(img2.astype('uint8'))
    def median_filter(self):
        rows, cols, channels = self.npimg.shape
        img2 = self.npimg.copy().astype('int64')
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                img2[i, j, :] = sorted(img2[i - 1:i + 2, j - 1:j + 2, 0].flatten())[4], \
                                sorted(img2[i - 1:i + 2, j - 1:j + 2, 1].flatten())[4], \
                                sorted(img2[i - 1:i + 2, j - 1:j + 2, 2].flatten())[4]
        return Image.fromarray(img2.astype('uint8'))
    def Guassion_filter(self, sigma, k=3):# 高斯分布
        img=self.npimg
        def get_guassion_arr():
            center = int(k / 2)
            arr = np.zeros((k, k)).astype(float)
            sum = 0.00
            for i in range(k):
                for j in range(k):
                    arr[i][j] = math.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma * sigma))
                    sum += arr[i][j]
            arr /= sum
            return arr
        center = int(k / 2)
        arr2 = get_guassion_arr()
        img2 = img.copy()
        rows, cols, channels = img.shape
        for i in range(center, rows - center):
            for j in range(center, cols - center):
                img2[i, j, :] = (img2[i - center:i + center + 1, j - center:j + center + 1, :] * arr2).sum(axis=0).sum(axis=0)
        return Image.fromarray(img2)
    def bilateral_filter(self,sigmad, sigmar, k=3):
        img2 = self.npimg.copy()
        rows, cols, channels = img2.shape
        center = int(k / 2)
        img2 = img2.astype('int64')

        def get_bilateral_arr(x, y, c):
            arr = np.zeros((k, k)).astype(float)
            sum = 0.00
            for i in range(k):
                for j in range(k):
                    arr[i][j] = math.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigmad * sigmad))
                    t = int(img2[i - center + x][j - center + y][c] - img2[x][y][c])
                    temp = math.exp(-(t) ** 2 / (2 * sigmar * sigmar))
                    arr[i][j] = arr[i][j] * temp
                    sum += arr[i][j]
            arr /= sum
            return arr
        for i in range(rows):
            for j in range(cols):
                for c in range(channels):
                    if i >= center and j >= center and i < rows - center and j < cols - center:
                        arr = get_bilateral_arr(i, j, c)
                        img2[i, j, c] = np.sum(img2[i - center:i + center + 1, j - center:j + center + 1, c] * arr)
        return Image.fromarray(img2.astype('uint8'))
