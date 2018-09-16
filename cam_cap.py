import cv2
import numpy as np
import ctypes
import sys

# initialize the camera
cam1 = cv2.VideoCapture(0)   # 0 -> index of camera
cam1.set(3, 2048)
cam1.set(4, 1536)
cam1.set(5, 20)

cam2 = cv2.VideoCapture(1)   # 0 -> index of camera
cam2.set(3, 1280)
cam2.set(4, 1024)
cam2.set(5, 40)

while True:
    ret1, img1 = cam1.read()
    ret2, img2 = cam2.read()
    img1 = cv2.flip(img1, 0)
    #img2 = cv2.flip(img2, 0)

    if ret1 and ret2:    # frame captured without any errors
        cv2.namedWindow("stereo_monitor", 2)
        cv2.imshow("cam1_pic",img1)
        cv2.imshow("cam2_pic", img2)
        combine = np.concatenate((img2, img1), axis=1) #combine image
        combine2 = cv2.resize(combine, (1280, 512), interpolation=cv2.INTER_AREA)
        cv2.imshow("stereo_monitor", combine2)

        #destroyWindow("cam1-test")
        # destroyWindow("cam2-test")

        k = cv2.waitKey(10) & 0xFF
        if k == 32:

            for i in range(3):
                cv2.waitKey(10)
                # imwrite('CO_0520_cam1_'+str(i)+'.jpg',img1) #save image1
                # imwrite('CO2_cam2_'+str(i)+'.jpg', img2)  # save image2
                cv2.imwrite('309_combine_pic' + str(i) + '.bmp', combine)
        if k == 27: break



    #cv2.destroyWindow("stereo_monitor")





