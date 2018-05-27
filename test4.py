import cv2
import numpy as np

def minfilter(img,r):
    H,W=img.shape
    im=img.copy()
    center=r
    for row in range(center,H-center):
        for col in range(center,W-center):
            im[row][col]=img[row-center:row+center+1,col-center:col+center+1].min(0).min(0)
    return im

def guidedfilter(I, p, r, eps):

    height, width= I.shape
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


def getV1(m, r, eps, w, maxV1):  # 输入rgb图像，值范围[0,1]

    V1 = np.min(m,2) # 得到暗通道图像
    #V1=np.expand_dims(V1,2)
    #img1 = np.expand_dims(minfilter(V1, 7),2)
    #cv2.imshow("4",img1)
    #cv2.imshow('h',minfilter(V1, 7))
    V1 = guidedfilter(V1,minfilter(V1, 7), r, eps)  # 使用引导滤波优化
    cv2.imshow("h",V1)
    #cv2.imshow("3",V1)
    #cv2.waitKey(0)
    bins = 3000
    ht = np.histogram(V1, bins)  # 计算大气光照A
    #统计V1max V1min 除以3000的各个区间内个数
    d = np.cumsum(ht[0]) / float(V1.size)
    #除以总的灰度值个数
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    print(d[lmax])#取的是亮度值
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    print(A)
    V1 = np.minimum(V1 * 0.95,0.8)  # 对值范围进行限制
    #print(V1[0][:10])


    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    #print(m.shape)
    V1, A = getV1(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
    print(A)
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    print(Y[0][:10])
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作'''
    return Y


if __name__ == '__main__':
    m = deHaze(cv2.imread('/home/wlj/pic/15.jpg')/255.0)*255
    m=np.clip(m,0,255)
    m=m.astype('uint8')
    cv2.imshow('defog.jpg', m)
    img=cv2.imread('/home/wlj/pic/15.jpg')
    #cv2.imshow("2",img)
    cv2.waitKey(0)
