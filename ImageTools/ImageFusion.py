from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img1=Image.open("/home/wlj/pic/8.jpg")
img2=Image.open("/home/wlj/pic/9.jpg")

img1=np.array(img1)
img2=np.array(img2)

alta=0.5

img3=img1*alta+img2*(1-alta)
img3[img3>255]=255
img3[img3<0]=0

img3=img3.astype('uint8')

plt.imshow(img3)
plt.show()