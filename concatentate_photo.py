import cv2
import numpy as np

for i in range(20):
    img1 = cv2.imread("calib/L{}.bmp".format(i))
    #img1 = cv2.flip(img1, 0)
    img2 = cv2.imread("calib/R{}.bmp".format(i))

    outputImg = np.concatenate((img1, img2), axis=1) #combine image
    cv2.imwrite("calib/combine" + str(i) + ".bmp", outputImg)

