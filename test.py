import numpy as np
dst=np.array([[5,8,12,3],[4,2,3,9],[7,3,4,2],[4,5,7,8]])
dst_min=dst.copy()
rows,cols=dst.shape
path_dict={}
for i in range(1,rows):
    for j in range(cols):
        dst_min[i,j]=dst[i,j]+min(dst_min[i-1,max(j-1,0)],dst_min[i-1,j],dst_min[i-1,min(j+1,cols-1)])
        t=np.where(dst_min[i - 1,max(j-1,0):min(j,cols-1) + 1]==np.min(dst_min[i - 1,max(j-1,0):min(j,cols-1) + 1]))
        print(i,dst_min[i - 1,max(j-1,0):min(j+1,cols-1) + 1],t[0])
        path_dict[(i,j)]=t
print(dst_min)
print(path_dict)
gggg

#根据最小值求路径
mask=(dst_min==dst_min)
minnum=np.min(dst_min[rows-1,:])
path_dict={}
j=0
for j in range(cols):
    if dst_min[rows-1,j]==minnum:
        mask[rows-1,j]=False
        break
for i in range(rows-1,0,-1):
    minnum=np.min(dst[i-1,max(0,j-1):min(cols-1,j+1)+1])
    l=len(mask[i-1,max(0,j-1):min(cols-1,j+1)+1])
    for k in range(l):
        if minnum==dst[i-1,max(0,j-1)+k]:
            mask[i - 1, max(0, j - 1)+k]=False
            j=max(0,j-1)+k
            break
print(path_dict)