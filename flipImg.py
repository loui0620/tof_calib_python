import cv2
import numpy as np


mat1 = np.array([1, 2])
mat2 = np.array([[1,3,2,4], [6,2,3,1]])
mat3 = np.array([[4], [3]])
#mat = np.cross(mat1, mat3)
print(np.shape(mat3))
"""
for i in range(4):
    img = cv2.imread("imgs/depthColor"+str(i)+".png")
    img = cv2.flip(img, 1)
    cv2.imwrite("imgs/depthFlipped"+str(i)+".png", img)


depth = cv2.imread("imgs/depthFlipped0.png", 0)
dWidth = 512
dHeight = 424
for ux in range(0, dWidth):
    for vy in range(0, dHeight):
        zValue = depth[vy, ux]# Element value represents depth val
"""