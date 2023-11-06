#!/usr/bin/env python
import cv2
import json
import numpy as np
import open3d as o3d

class Template:
    def __init__(self, imgPath, imgName, R, t, K, D, vertices, height, width, iterationsCount=1000, reprojectionError=5.0, display=False, debug=False,contrastThreshold=0.04, edgeThreshold=10.0):

        self.imgPath = imgPath
        self.vertices = vertices
        self.imgName = imgName
        baseImg = cv2.imread(imgPath, cv2.IMREAD_COLOR)
        baseImg = cv2.undistort(baseImg, K, D)
        self.img = baseImg
        self.display = display
        self.debug = debug
        self.iterationsCount = iterationsCount
        self.reprojectionError = reprojectionError
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.K = K
        self.D = D
        if self.display:
            self.show()
        imagePoints, _ = cv2.projectPoints(vertices, R, t, K, D)
        # 将投影后的点坐标和深度信息转换为二维数组
        imagePoints = imagePoints.reshape(-1, 2)
        pointMap = np.zeros((height, width, 3), dtype=np.float64)
        cnt = np.zeros((height, width), dtype=np.float64)
        if self.display:
            img2 = baseImg.copy()
        P = K @ np.hstack((R, t))
        for j in range(len(imagePoints)):
            x = int(imagePoints[j][0])
            y = int(imagePoints[j][1])
            if cnt[y][x]>1:
                temp = vertices[j]
                now = pointMap[y][x]
                pt = P @ np.hstack((temp, 1))
                pn = P @ np.hstack((now, 1))
                if (pt[2]<pn[2]):
                    pointMap[y][x] = vertices[j]
            else:
                pointMap[y][x] = vertices[j]
            cnt[y][x] += 1
            if display:
                cv2.circle(img2, (int(imagePoints[j][0]), int(imagePoints[j][1])), 0, (0, 255, 0), 1)
            
        if self.display:
            cv2.imshow("Image", img2)
            cv2.imshow("pointMap", pointMap)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()
        self.pointMap = pointMap
        self.cnt = cnt    
    def show(self):
        show = cv2.resize(self.img, (0,0), fx=0.3, fy=0.3)
        cv2.imshow(self.imgName, show)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()
    def match(self, targetImg, display=False, debug=False):
        baseImg = self.img
        # 创建SIFT对象
        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=self.contrastThreshold, edgeThreshold=self.edgeThreshold)
        # 检测关键点并计算描述符
        kp1, des1 = sift.detectAndCompute(baseImg, None)
        kp2, des2 = sift.detectAndCompute(targetImg, None)

        # 创建BFMatcher对象
        bf = cv2.BFMatcher()

        # 使用BFMatcher对象匹配关键点
        matches = bf.match(des1, des2)

        # 将匹配结果按照距离排序
        matches = sorted(matches, key=lambda x: x.distance)
        if self.debug:
            print('matches:', len(matches))
            
        # 显示匹配结果
        if display:
        
            img3 = cv2.drawMatches(baseImg, kp1, targetImg, kp2, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('Matches', cv2.resize(img3, (0,0), fx=0.5, fy=0.5))
            cv2.waitKey(1)
            # cv2.destroyAllWindows()
        
        basePoints2d = []
        targetPoints2d = []
        for i in range(len(matches)):
            basePoints2d.append(kp1[matches[i].queryIdx].pt)
            targetPoints2d.append(kp2[matches[i].trainIdx].pt)
        
        points2d = []
        points3d = []
        for i in range(len(targetPoints2d)):
            x = int(basePoints2d[i][0])
            y = int(basePoints2d[i][1])
            if self.cnt[y][x]>0:
                points2d.append(targetPoints2d[i])
                points3d.append(self.pointMap[y][x])
        if debug:
            print('2D and 3D points pairs:', len(points2d))
        points2d = np.array(points2d, dtype=np.float32)
        points3d = np.array(points3d, dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(points3d, points2d, self.K, self.D, iterationsCount=self.iterationsCount, reprojectionError=self.reprojectionError)
        R_est = None
        t_est = None
        if success:
            R_est, _ = cv2.Rodrigues(rvec)
            t_est = tvec
            if debug:
                print('solvePnPRansac success')
                print('inliers:', len(inliers))
                print('R_est:', R_est)
                print('t_est:', t_est)
        else:
            if debug:
                print('solvePnPRansac failed')
        
        points2d = points2d[inliers]
        points3d = points3d[inliers]
        points2d = points2d.reshape(-1, 2)
        points3d = points3d.reshape(-1, 3)
        
        # 重投影
        rePoints2d, _ = cv2.projectPoints(points3d, R_est, t_est, self.K, self.D)  
        # 将投影后的点坐标和深度信息转换为二维数组
        rePoints2d = rePoints2d.reshape(-1, 2)
        
        error = np.mean(np.abs(points2d-rePoints2d))
        if debug:
            print('reprojection error:', error)
        if display:
            self.drawReprojection(targetImg, points2d, rePoints2d)
            self.drawReprojectionAll(targetImg, self.vertices, R_est, t_est, self.K, self.D)
        return success, R_est, t_est, error
    def drawReprojection(self, img,points2d, rePoints2d):
        img2 = img.copy()
        for i in range(len(points2d)):
            cv2.circle(img2, (int(points2d[i][0]), int(points2d[i][1])), 1, (0, 255, 0), 1)
            cv2.circle(img2, (int(rePoints2d[i][0]), int(rePoints2d[i][1])), 1, (0, 0, 255), 1)
        cv2.imshow("Reprojection", img2)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()
    def drawReprojectionAll(self,img, points3d, R, t, K, D):
        img = img.copy()
        points2d,_ = cv2.projectPoints(points3d, R, t, K, D)
        points2d = points2d.reshape(-1, 2)
        for i in range(len(points2d)):
            cv2.circle(img, (int(points2d[i][0]), int(points2d[i][1])), 0, (0, 255, 0), 1)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()
        
# def getVertices(pointPath):
#     pointCloud = o3d.io.read_triangle_mesh(pointPath)
#     # 将点云转换为NumPy数组
#     vertices = np.asarray(pointCloud.vertices)
#     print("Load {} points".format(vertices.shape[0]))
#     return vertices

# def loadTemplates(templatePath):
#     jsonPath = templatePath + 'poses.json'
#     poseJson = json.load(open(jsonPath))
#     imgNames = list(poseJson.keys())
#     pointPath = templatePath + 'model/' + poseJson[imgNames[0]][0]['obj']
#     vertices = getVertices(pointPath)
#     K = np.array([1077.33, 0.0, 911.133, 0.0, 1076.2, 530.144, 0.0, 0.0, 1.0]).reshape(3, 3)
#     # D = np.array([-0.0408,0.0098,-0.0006,-00.0009,-0.0046])
#     D = None
#     height = 1080
#     width = 1920
#     iterationsCount = 1000
#     reprojectionError = 100.0
#     print('Load {} templates'.format(len(imgNames)))
#     templates = []
#     imgName = imgNames[0]
#     imgPath = templatePath + imgName
#     R = np.array(poseJson[imgName][0]['R']).reshape(3,3)
#     t = np.array(poseJson[imgName][0]['t']).reshape(3,1)
#     T = np.hstack((R, t))
#     T = np.vstack((T, np.array([0,0,0,1])))
#     for imgName in imgNames:
#         imgPath = templatePath + 'img/' + imgName
#         R = np.array(poseJson[imgName][0]['R']).reshape(3,3)
#         t = np.array(poseJson[imgName][0]['t']).reshape(3,1)
#         T = np.hstack((R, t))
#         T = np.vstack((T, np.array([0,0,0,1])))
#         templates.append(Template(imgPath, imgName, R, t, K, D, vertices, height, width, iterationsCount, reprojectionError, display=False, debug=False,contrastThreshold=0.04, edgeThreshold=10.0))
#     return templates
    
# def matchAll(img, templates, debug=False, display=False):
#     R_best = None
#     t_best = None
#     error_best = None
#     success_best = False
#     id_best = None
#     for i in range(len(templates)):
#         template = templates[i]
#         success, R_est, t_est, error = template.match(img, display=display, debug=debug)
#         if success:
#             if not success_best:
#                 success_best = True
#                 R_best = R_est
#                 t_best = t_est
#                 error_best = error
#                 id_best = i
#             else:
#                 if error<error_best:
#                     R_best = R_est
#                     t_best = t_est
#                     error_best = error
#                     id_best = i
#     if success_best:
#         if debug:
#             print('id_best:', id_best)
#             print('R_best:', R_best)
#             print('t_best:', t_best)
#             print('error_best:', error_best)
#         return True, R_best, t_best, error_best
#     else:
#         return False, None, None, None
if __name__ == '__main__':
    templatePath = '../../templates/'
    templates = loadTemplates(templatePath)
    targetPath = '/home/crab2/zsy/abs_loc/data/point2depthgraph/left001965.png'
    targetImg = cv2.imread(targetPath, cv2.IMREAD_COLOR)
    print(matchAll(targetImg, templates, debug=True, display=True))