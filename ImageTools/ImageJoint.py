import cv2
import numpy as np

MIN_MATCH_COUNT = 10
img1 = cv2.imread('/home/wlj/pic/8.jpg')          # queryImage
img2 = cv2.imread('/home/wlj/pic/9.jpg')          # trainImage

def SIFT():
    # 初始化特征检测器
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img2,None)
    kp2, des2 = sift.detectAndCompute(img1,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
          good.append(m)
    # cv2.drawMatchesKnn expects list of lists as matches.
    #good_2 = np.expand_dims(good, 1)
    #matching=cv2.drawMatchesKnn(img1,kp1,img2,kp2,good_2[:20],None,flags=2)

    if len(good)>MIN_MATCH_COUNT:
        # 获取关键点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts,cv2.RANSAC,5.0)
        wrap = cv2.warpPerspective(img2, H, (img1.shape[1]+img2.shape[1] , img1.shape[0]+img2.shape[0]))
        #print(wrap.shape)
        wrap[0:img2.shape[0], 0:img2.shape[1]] = img1
        #print(img1.shape,img2.shape)
        rows, cols = np.where(wrap[:,:,0] !=0)
        min_row, max_row = min(rows), max(rows) +1
        min_col, max_col = min(cols), max(cols) +1
        result = wrap[min_row:max_row,min_col:max_col,:]#去除黑色无用部分

        return result
        return wrap

if __name__ == '__main__':
    result = SIFT()
    #cv2.imshow('img3.jpg',matching)
    cv2.imshow('result.jpg',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)