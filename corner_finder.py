import cv2
import numpy as np
import logging

class cornerFinder():
    def __init__(self, cornerSize, winSize):
        self.FAST = cv2.FastFeatureDetector()
        self.cornerSize = cornerSize
        self.winSize = winSize
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30 , 0.001)
        corner_x = []
        corner_p = []

    def findKeyPoint(self, img):
        if img is None:
            logging.warning("No images found for corner detection.")
            return None
        h = img.shape[0]
        w = img.shape[1]

        found, corners = cv2.findChessboardCorners(img, self.cornerSize)
        if found:
            cv2.cornerSubPix(img, corners, self.winSize, (-1,-1), self.criteria)

        if not found:
            logging.warning("chessboard not found.")
            return None

    def printFastParams(self):
        print("Threshold: ", self.FAST.getInt('threshold'))
        print("nonMaxSupression: ", self.FAST.getBool('nonmaxSupression'))
        print("neighborhood")