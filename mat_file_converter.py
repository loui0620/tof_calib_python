import scipy.io as sio
import numpy as np
import os
from os.path import join


class loadMatFile():
    def __init__(self):
        self.content = None
        self.struct = None

    def loadMat(self, filePath):
        self.content = sio.loadmat(filePath)
        #self.struct = sio.matlab(filePath)

    def getDepthRotate(self):
        dR = self.content['dR']
        dR = np.array(dR)
        return dR

    def getDepthTranslate(self):
        dt = self.content['dt']
        dt = np.array(dt)
        return dt

    def getDepthIntrinsic(self):
        dK = self.content['dK']
        dK = np.array(dK)
        return dK

    def getColorIntrinsic(self):
        rK = self.content['rK']
        rK = np.array(rK)
        return rK

    def getDepthCorner(self, t):
        # depth_corner_p: projected coordinate on camera plane.
        # depth_corner_x: given coordinate on chessboard plane.
        depth_corner_p = self.content['depth_corner_p']
        projected_list = []
        for i in range(t):
            projected_point = depth_corner_p[0,i]
            projected_point = np.transpose(projected_point)
            projected_list.append(projected_point)
        #projected_list = np.array(projected_list)
        
        depth_corner_x = self.content['depth_corner_x']
        depth_corner_x = np.array(depth_corner_x)
        depth_corner_x.reshape([-1, 2])

        return projected_list, depth_corner_x

    def getHomographyOfMatlab(self, t):
        dH = self.content['dH']
        H_list = []
        for i in range(t):
            H = dH[0, i]
            H_list.append(H)

        return H_list

    def getTvecDepthFromMatlab(self, t):
        tvec_depth = self.content['t0']
        tvec_list = []
        for i in range(t):
            tvec = tvec_depth[:, i]
            tvec_list.append(tvec)
        
        tvec_list = np.asarray(tvec_list)
        return tvec_list

    def getTransformationFromCalib0(self, t):
        RextRaw = self.content['Rext']# take index element from cell-list
        textRaw = self.content['text']

        RextList = []
        textList = []

        for i in range(t):
            Rext = RextRaw[0,i]
            text = textRaw[0,i]
            RextList.append(Rext)
            textList.append(text)

        RextList = np.array(RextList)
        textList = np.array(textList)
        return RextList, textList

