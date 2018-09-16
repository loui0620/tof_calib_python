import cv2
import numpy as np
from mat_file_converter import loadMatFile
from Extrinsic_extraction import ymlParser

runPointCloud2Color = True
runDepth2Color = False
runPointCloud2Depth = False

pId = 4665
ymlPath = "imgs/YMLs/calib.yml"

depthFile = "D:\ReconstructionEngineSource\ReconstructionEngine\working/4/4596\depth/depth_100.png"
colorFile = "D:\ReconstructionEngineSource\ReconstructionEngine\working/4/4596\source/color_1_02.jpg"
#meshName = "imgs/pointCloudFile/PCMesh104.obj"

depthImg = cv2.imread(depthFile, 0)
colorImg = cv2.imread(colorFile)

depthYML = cv2.FileStorage(ymlPath, cv2.FILE_STORAGE_READ)
depthMtx = depthYML.getNode("depthValue").mat()
depthMtx = np.array(depthMtx)
depthMtx = depthMtx.transpose()
depthImg = np.array(depthImg)

# depthImg = np.uint16(depthImg) # convert from 32 bit to 16 bit

ymlLoader = ymlParser
ymlLoader.loadFile(ymlPath)
dK = ymlLoader.getDepthIntrinsic()
rK = ymlLoader.getRGBIntrinsic()
dR = ymlLoader.getRotationExtrinsic()
dt = ymlLoader.getTranslationExtrinsic()


depthHeight, depthWidth = depthImg.shape[0], depthImg.shape[1]
colorHeight, colorWidth = colorImg.shape[0], colorImg.shape[1]

depthMap = np.zeros((colorHeight, colorWidth))
depthRaw = np.zeros((depthHeight, depthWidth))
factor = 1 # for 16 bit float images
dfx, dfy, dcx, dcy = dK[0, 0], dK[1, 1], dK[0, 2], dK[1, 2]
cfx, cfy, ccx, ccy = rK[0, 0], rK[1, 1], rK[0, 2], rK[1, 2]
print(dfx, dfy, dcx, dcy)

scale_factor = 0.018257575
scale_factor = 0.01

def projectPointCloudTo2D(pt3d, focal_x, focal_y, center_x, center_y):
    pt2d = np.zeros((2,1))
    pt2d[0, 0] = project1D(pt3d[0, 0], pt3d[2, 0], focal_x, center_x)
    pt2d[1, 0] = project1D(pt3d[1, 0], pt3d[2, 0], focal_y, center_y)
    return pt2d


def project1D(pt, depth, focal, center):
    ret = (pt * focal / depth) + center
    return ret


def computeDepthProjection(depth, color):
    dHeight, dWidth = depth.shape[0], depth.shape[1]
    cHeight, cWidth = color.shape[0], color.shape[1]
    depthTarget = np.zeros((cHeight, cWidth))

    colorMap = np.zeros((cHeight, cWidth, 3), np.uint8)

    pts = []
    for ux in range(0, dWidth):
        for vy in range(0, dHeight):
            zValue = depth[vy, ux] * scale_factor# Element value represents depth val
            if zValue > 0:
                print(zValue)
                xk = (ux - dcx) / dfx
                yk = (vy - dcy) / dfy
                Z = zValue
                X = xk * Z
                Y = yk * Z

                pt = np.array([X, Y, Z])
                pt = np.dot(dR, pt)  # TODO:1. transpose or????
                pt = np.add(pt, dt.transpose())  # TODO:2. transpose or????

                xk = pt[0, 0] / pt[0, 2]
                yk = pt[0, 1] / pt[0, 2]

                xc = (cfx * xk) + ccx  # TODO:3. multiply 0.5 in EnvTool
                yc = (cfy * yk) + ccy

                if yc > 0:
                    xc = int(xc)
                    yc = int(yc)

                    if xc < cWidth and yc < cHeight:
                        # cv2.circle(depthTarget, (xc, yc), 2, (55,255,55), -1)
                        cv2.circle(colorMap, (xc, yc), 2, (55, 255, 0), -1)
                    pts.append(pt)
    cv2.imwrite("imgs/colorMap.png", colorMap)
    color = np.asarray(color)
    depthTarget = depthTarget.astype(float)
    colorMap = colorMap.astype(float)
    color = color.astype(float)
    overlapping = cv2.addWeighted(color, 0.5, colorMap, 0.5, 0)
    return overlapping


class ObjLoader(object):
    def __init__(self, fileName):
        self.vertices = []
        self.faces = []
        ##
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    vertex = (round(vertex[0], 3), round(vertex[1], 3), round(vertex[2], 3))
                    vertex = np.array([[vertex[0]], [vertex[1]], [vertex[2]]])
                    self.vertices.append(vertex)
                    print(vertex[2])

                elif line[0] == "f":
                    string = line.replace("//", "/")
                    ##
                    i = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            face.append(string[i:-1])
                            break
                        face.append(string[i:string.find(" ", i)])
                        i = string.find(" ", i) + 1
                    ##
                    self.faces.append(tuple(face))

            f.close()

            print(fileName, "loaded.")
        except IOError:
            print(fileName, " file not found.")

    def getModelVertices(self):
        vertices = self.vertices
        return vertices


def computePointCloudToDepth(meshFile, depthImg):
    meshLoader = ObjLoader(meshFile)
    mesh = meshLoader.getModelVertices()

    projectedPts = []
    projectedImg = np.zeros((depthHeight, depthWidth, 3), np.uint8)

    for i in range(len(mesh)):
        pt3d = mesh[i]
        projected = projectPointCloudTo2D(pt3d, dfx, dfy, dcx, dcy)
        if projected[0, 0] < depthWidth and projected[1, 0] < depthHeight:
            cv2.circle(projectedImg, (int(projected[0,0]), int(projected[1, 0])), 1, (55,255,0), -1)
            projectedPts.append(projected)

    projectedImg = cv2.flip(projectedImg, 1)
    cv2.imwrite("imgs/pt2depth.png", projectedImg)
    overlapped = cv2.addWeighted(projectedImg, 0.4, depthImg, 0.6, 0)

    return overlapped


meshNameList = np.array([0,17,48,83,111,150,177,206,245,276,308,348,380,416,450,481,515,548,575])
# meshNameList = np.array([0,46,104,145,192,264,307,380,419,450,512,542,587,649,692,755,796,834,894])

"""
#######PointCloud 2 Depth###########
depthOut = cv2.imread(depthFile)
ret = computePointCloudToDepth(meshName, depthOut)
cv2.imwrite("imgs/ret.png", ret)

#######Depth 2 Color#############
over = computeDepthProjection(depthImg, colorImg)
cv2.imwrite('imgs/overlapped_Imgs/overlapping1.png', over)
print("02 Done!!")
"""
if runPointCloud2Color == True:
    for m in range(1):
        meshStr = "D:\ReconstructionEngineSource\ReconstructionEngine\working/4/"+str(pId)+"\source\PCMesh_"+str(meshNameList[m])+".obj"
        if m < 10:
            depthName = "D:\ReconstructionEngineSource\ReconstructionEngine\working/4/"+str(pId)+"\depth/depth_0"+str(m)+".png"
            colorName = "D:\ReconstructionEngineSource\ReconstructionEngine\working/4/"+str(pId)+"\source\color_0"+str(m)+".jpg"
        elif m >= 10:
            depthName = "D:\ReconstructionEngineSource\ReconstructionEngine\working/4/"+str(pId)+"\depth/depth_"+str(m)+".png"
            colorName = "D:\ReconstructionEngineSource\ReconstructionEngine\working/4/"+str(pId)+"\source\color_"+str(m)+".jpg"

        blankImg = np.zeros((depthHeight, depthWidth, 3), np.uint8)
        meshImage = computePointCloudToDepth(meshStr, blankImg)
        cv2.imwrite("imgs/"+str(pId)+"/mesh2D"+str(m)+".png", meshImage)

        colorPic = cv2.imread(colorName)
        depthPic = cv2.imread(depthName, 0)
        over = computeDepthProjection(depthPic, colorPic)
        #cv2.imwrite('imgs/' + str(pId) + '/pointcloud2depth_' + str(m) + '.png', over)
        print(str(m), " depth2plane image projected.")

if runDepth2Color == True:
    for k in range(4):
        if k < 10:
            depthName = "D:\ReconstructionEngineSource\ReconstructionEngine\working/4/"+str(pId)+"\depth/depth_0"+str(k)+".png"
            colorName = "D:\ReconstructionEngineSource\ReconstructionEngine\working/4/"+str(pId)+"\source\color_0"+str(k)+".jpg"
        elif k >= 10:
            depthName = "D:\ReconstructionEngineSource\ReconstructionEngine\working/4/"+str(pId)+"\depth/depth_"+str(k)+".png"
            colorName = "D:\ReconstructionEngineSource\ReconstructionEngine\working/4/"+str(pId)+"\source\color_"+str(k)+".jpg"

        depthName = "imgs/4665/mesh2D"+str(k)+".png"
        depthPic = cv2.imread(depthName, 0)
        if depthPic is not None:
            print("depth image ", k, " loaded.")
        colorPic = cv2.imread(colorName)
        over = computeDepthProjection(depthPic, colorPic)
        cv2.imwrite('imgs/'+str(pId)+'/pointcloudTo4k_'+str(k)+'.png', over)

if runPointCloud2Depth == True:
    for i in range(19):
        if i < 10:
            depthName = "imgs/depth/depth_0"+str(i)+".png"

        elif i >= 10:
            depthName = "imgs/depth/depth_"+str(i)+".png"

        depthPNG = cv2.imread(depthName, 1)
        meshName = "imgs\pointCloudFile\PCMesh"+str(meshNameList[i])+".obj"
        retImg = computePointCloudToDepth(meshName, depthPNG)
        ret4k = computeDepthProjection(depthPNG, colorImg)
        cv2.imwrite("imgs/depth_pontcloud_mapping/ret" + str(i) + ".png", retImg)
        i += 1
"""
meshLoader = ObjLoader("imgs/PCMesh40.obj")
mesh = meshLoader.getModelVertices()

pts = []
pointCloud2D = np.zeros((depthHeight, depthWidth, 3), np.uint8)
for p in range(len(mesh)):
    pt3d = mesh[p]
    ret2d = projectPointCloudTo2D(pt3d, dfx, dfy, dcx, dcy)
    if ret2d[0,0] < depthWidth and ret2d[1,0] < depthHeight:
        cv2.circle(pointCloud2D, (int(ret2d[0,0]), int(ret2d[1,0])), 1, (55, 255, 55), -1)
    pts.append(ret2d)
pointCloud2D = cv2.flip(pointCloud2D, 1)
# pointCloud2D = cv2.resize(pointCloud2D, (2048, 1696),  interpolation=cv2.INTER_LINEAR)
cv2.imwrite("imgs/pc2d.jpg", pointCloud2D)
"""


