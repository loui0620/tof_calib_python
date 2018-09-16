import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mat_file_converter import loadMatFile

# implementation of KinectCALIB_MATLAB visualization modules
# testing for cv2 importing.

valid_idx = 30
depthNormal_list = []
depthDistance_list = []
depthTranslate_list = []

# 8 IR reflective balls
x= np.array([[0,0,0.295,0.59,0.59,0.59,0.295,0],
         [0.2175,0.435,0.435,0.435,0.2175,0,0,0]])

loader = loadMatFile()
loader.loadMat("reference/KiectCalib_MATLAB/calibN_0622.mat")
R0, t0 = loader.getTransformationFromCalib0(valid_idx)
tvec_depth = loader.getTvecDepthFromMatlab(valid_idx)

homography_list = loader.getHomographyOfMatlab(valid_idx)# current: DEPTH is jump over the cv2.findHomography(). TODO: depth computing from "depth_corner_p"
dK = loader.getDepthIntrinsic()

class ResultContainer:
    def __init__(self):
        self.calib0 = {}

class calibImages:
    def __init__(self):
        self.datasetPath = ""
        self.depthFiles = []
        self.rgbFiles = []
        self.bmpFiles = []


class ymlParser:
    def loadFile(self, ymlPath):
        self.fs = cv2.FileStorage(ymlPath, cv2.FILE_STORAGE_READ)

    def getCamId(self):
        camId = self.fs.getNode("camID")
        return camId

    def getRSize(self):
        rSize1 = self.fs.getNode("rsize1")
        return rSize1

    def getRGBIntrinsic(self):
        rK1 = self.fs.getNode("rK1")
        rK1 = np.array(rK1.mat())
        return rK1

    def getDepthIntrinsic(self):
        dK = self.fs.getNode("dK")
        dK = np.array(dK.mat())
        return dK

    def getRotationExtrinsic(self):
        dR = self.fs.getNode("dR")
        dR = np.array(dR.mat())
        return dR

    def getTranslationExtrinsic(self):
        dt = self.fs.getNode("dt")
        dt = np.array(dt.mat())
        return dt

def extrinsic2plane(Rext, text):
    N = Rext[:,2]
    d = np.dot(Rext[:,2], text)

    return N, d

def getColorNormals(Rext, text):
    normalList = []
    distanceList = []
    
    for i in range(valid_idx):
        rplaneN, rplaned = extrinsic2plane(Rext[i], text[i])
        normalList.append(rplaneN)
        distanceList.append(rplaned)
    return normalList, distanceList

def getHomographyFromCorners(depth_corner_x, depth_corner_p):

    dH, depth_plane_mask = cv2.findHomography(depth_corner_x, depth_corner_p, cv2.RANSAC, 5.0)
    matchesMask = depth_plane_mask.ravel().tolist()
    return dH, matchesMask

def getExternFromHomography(dK, H):
    planeRotate = np.zeros((3, 3))
    test = np.linalg.inv(dK)
    planeRotate[:, [0]] = np.dot(np.linalg.inv(dK), H[:, [0]])
    planeRotate[:, [1]] = np.dot(np.linalg.inv(dK), H[:, [1]])

    Lambda = 1 / np.linalg.norm(planeRotate[:, 0])
    planeRotate[:, [0]] = Lambda * planeRotate[:, [0]]
    planeRotate[:, [1]] = Lambda * planeRotate[:, [1]]
    planeRotate[:, 2] = np.cross(planeRotate[:, 0], planeRotate[:,1])

    u, s, vh = np.linalg.svd(planeRotate)
    
    R = np.dot(u, vh)
    planeTranslate = Lambda * (np.dot(np.linalg.inv(dK), H[:, [2]]))
    if planeTranslate[2, 0] < 0:
        planeTranslate = -1 * planeTranslate

    return R, planeTranslate

projectedCoordinate, chessCoordinate = loader.getDepthCorner(valid_idx)

colorNormal_list, colorDistance_list = getColorNormals(R0, t0)

# for debugging
p = np.array([[239,251,282,314,309,303,262,225],
          [231,253,255,257,235,206,206,205]])
x= np.array([[0,0,0.295,0.59,0.59,0.59,0.295,0],
         [0.2175,0.435,0.435,0.435,0.2175,0,0,0]])

p = np.transpose(p)
x = np.transpose(x)


for i in range(valid_idx):
    #H, projected_mask = getHomographyFromCorners(x, projectedCoordinate[i])
    H = homography_list[i] 
    depthRotate, depthTranslate = getExternFromHomography(dK, H)
    depthN, depthD = extrinsic2plane(depthRotate, depthTranslate)
    depthN = -1 * depthN # FIX VECTOR SIGN
    depthTranslate_list.append(depthTranslate)
    depthNormal_list.append(depthN)
    depthDistance_list.append(depthD)
    homography_list.append(H)

# coverting and transpose normal_stacks
depthNormal_list = np.asarray(depthNormal_list)
colorNormal_list = np.asarray(colorNormal_list)

depthDistance_list = np.asarray(depthDistance_list)
colorDistance_list = np.asarray(colorDistance_list)

depthTranslate_list = np.asarray(depthTranslate_list)
colorTranslate_list = np.asarray(t0)


""" COMPUTE EXTRINSIC PARAMATERS """
# get dt: depth2color translation
dt_pre = np.linalg.inv(np.dot(colorNormal_list.transpose(), colorNormal_list)) 
dt_mid = (colorDistance_list - depthDistance_list)
dt_post = np.dot(colorNormal_list.transpose(), dt_mid)
dt = np.dot(dt_pre, dt_post)
dt = dt * -1 # FIX VECTOR SIGN

# get dR: depth2color rotation
u, s, v = np.linalg.svd(np.dot(depthNormal_list.transpose(), colorNormal_list))
dR = np.dot(v, u)

if np.linalg.det(dR) < 0:
     dR = -1 * dR

extrinsic_matrix = np.concatenate((dR, dt), axis = 1)
""" WRITE YML FILES """

# print("dR: \n", dR)
# print("dt: \n", dt)
print("Final: \n", extrinsic_matrix)

""" PLOT 3D NORMAL VECTOR """
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_aspect(1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

xColor = colorTranslate_list[:, 0, 0]
yColor = colorTranslate_list[:, 1, 0]
zColor = colorTranslate_list[:, 2, 0]
uColor = colorNormal_list[:, 0]
vColor = colorNormal_list[:, 1]
wColor = colorNormal_list[:, 2]

# if depthTranslate_list[:, 2, 0].any() < 0:
xDepth = depthTranslate_list[:, 0, 0]
yDepth = depthTranslate_list[:, 1, 0]
zDepth = depthTranslate_list[:, 2, 0]

uDepth = depthNormal_list[:, 0]
vDepth = depthNormal_list[:, 1]
wDepth = depthNormal_list[:, 2]

diff = colorTranslate_list[0, :, 0] - depthTranslate_list[0, :, 0]
print(np.linalg.norm(diff))
ax.quiver(xColor, yColor, zColor, uColor, vColor, wColor, length=0.1, normalize=True, color='r')
ax.quiver(xDepth, yDepth, zDepth, uDepth, vDepth, wDepth, length=0.1, normalize=True, color='b')
# plt.show()

ymlPath = "reference/calib.yml"
ymlParser = ymlParser()
ymlConfig = ymlParser.loadFile(ymlPath)
