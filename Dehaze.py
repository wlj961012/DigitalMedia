import numpy as np
from PIL import Image
import cv2

img=Image.open("/home/wlj/pic/7.png")
img=np.array(img)/255.0


def minfilter(img,r):
    H,W=img.shape
    im=img.copy()
    center=r
    for row in range(center,H-center):
        for col in range(center,W-center):
            im[row][col]=img[row-center:row+center+1,col-center:col+center+1].min(0).min(0)
    return im

def guardedfilter(I,p,r,eps):
    '''meanI=average_filter(I,r)
    meanp=average_filter(p,r)
    corrI=average_filter(I*I,r)
    corrIp=average_filter(I*p,r)
    varI=corrI-meanI*meanI
    covIp=corrIp-meanI*meanp
    a=covIp/(varI+eps)
    b=meanp-a*meanI
    meana=average_filter(a,r)
    meanb=average_filter(b,r)
    return meana*I+meanb'''
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))

    return m_a * I + m_b

def getI_A(imgg):
    img=imgg.copy()
    img1 = np.min(img, 2)
    img2 = minfilter(img1, 7)
    img2 = guardedfilter(img1, img2, 81, 0.001)
    bins = 3000
    ht = np.histogram(img2, bins)
    d = np.cumsum(ht[0]) / float(img2.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    print(d[lmax])
    A = np.mean(img, 2)[img2 >= ht[1][lmax]].max()
    print(A)
    img2*=0.95
    return img2,A

img2,A=getI_A(img)
Y=np.zeros(img.shape)
t0=np.zeros(img2.shape)
t0+=0.1
tx=np.maximum(1-img2,0.1)
for k in range(3):
    Y[:, :, k] = (img[:,:,k]-A)/np.maximum(1-img2,0.1)+A
# 颜色校正
Y = np.clip(Y, 0, 1)
print(Y[0][:10])
Y=Y*255
Y=Y.astype('uint8')

Image.fromarray(Y).show()