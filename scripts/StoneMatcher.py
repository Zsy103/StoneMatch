import Template
import numpy as np
import json
import open3d as o3d
import cv2
class StoneMatcher:
    def __init__(self, templatePath, debug=False, displayBest=False, displayAll=False):
        self.templatePath = templatePath
        self.templates = self.loadTemplates()
        self.debug = debug
        self.displayBest = displayBest
        self.displayAll = displayAll
    def getVertices(self,pointPath):
        pointCloud = o3d.io.read_triangle_mesh(pointPath)
        # 将点云转换为NumPy数组
        vertices = np.asarray(pointCloud.vertices)
        print("Load {} points".format(vertices.shape[0]))
        return vertices
    def loadTemplates(self):
        jsonPath = self.templatePath + 'poses.json'
        poseJson = json.load(open(jsonPath))
        imgNames = list(poseJson.keys())
        pointPath = self.templatePath + 'model/' + poseJson[imgNames[0]][0]['obj']
        vertices = self.getVertices(pointPath)
        K = np.array([1077.33, 0.0, 911.133, 0.0, 1076.2, 530.144, 0.0, 0.0, 1.0]).reshape(3, 3)
        # D = np.array([-0.0408,0.0098,-0.0006,-00.0009,-0.0046])
        D = None
        height = 1080
        width = 1920
        iterationsCount = 1000
        reprojectionError = 100.0
        print('Load {} templates'.format(len(imgNames)))
        templates = []
        imgName = imgNames[0]
        imgPath = self.templatePath + imgName
        R = np.array(poseJson[imgName][0]['R']).reshape(3,3)
        t = np.array(poseJson[imgName][0]['t']).reshape(3,1)
        T = np.hstack((R, t))
        T = np.vstack((T, np.array([0,0,0,1])))
        for imgName in imgNames:
            imgPath = self.templatePath + 'img/' + imgName
            R = np.array(poseJson[imgName][0]['R']).reshape(3,3)
            t = np.array(poseJson[imgName][0]['t']).reshape(3,1)
            T = np.hstack((R, t))
            T = np.vstack((T, np.array([0,0,0,1])))
            templates.append(Template.Template(imgPath, imgName, R, t, K, D, vertices, height, width, iterationsCount, reprojectionError, display=False, debug=False,contrastThreshold=0.04, edgeThreshold=10.0))
        return templates
    def matchAll(self, img):
        R_best = None
        t_best = None
        error_best = None
        success_best = False
        id_best = None
        templates = self.templates
        display = self.displayAll
        debug = self.debug
        for i in range(len(templates)):
            template = templates[i]
            success, R_est, t_est, error = template.match(img, display=display, debug=debug)
            if success:
                if not success_best:
                    success_best = True
                    R_best = R_est
                    t_best = t_est
                    error_best = error
                    id_best = i
                else:
                    if error<error_best:
                        R_best = R_est
                        t_best = t_est
                        error_best = error
                        id_best = i
        if self.displayBest:
            self.matchId(img, id_best)
        if success_best:
            if debug:
                print('id_best:', id_best)
                print('R_best:', R_best)
                print('t_best:', t_best)
                print('error_best:', error_best)
            return True, R_best, t_best, error_best, id_best
        else:
            return False, None, None, None, None
    def matchId(self, img, id):
        template = self.templates[id]
        display = self.displayBest
        debug = self.debug
        success, R_est, t_est, error = template.match(img, display=display, debug=debug)
        if success:
            if debug:
                print('id:', id)
                print('R:', R_est)
                print('t:', t_est)
                print('error:', error)
            return True, R_est, t_est, error
        else:
            return False, None, None, None

if __name__ == '__main__':
    stoneMatcher = StoneMatcher('../templates/', debug=True, displayBest=True, displayAll=False)
    targetPath = '/home/crab2/zsy/abs_loc/data/point2depthgraph/left001965.png'
    targetImg = cv2.imread(targetPath, cv2.IMREAD_COLOR)
    print(stoneMatcher.matchAll(targetImg))