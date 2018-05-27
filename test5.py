import numpy as np

C=np.array([[1,2],[3,4]])

C=np.pad(C, ((1, 1), (1, 1)), 'maximum')#上下左右

print(C)