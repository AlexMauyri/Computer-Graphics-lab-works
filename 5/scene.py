import numpy as np
from objmodel import ObjModel
from PIL import Image
from math import sin, cos, radians

class Scene:
    def __init__(self, height, width, light, camera):
        self.height = height
        self.width = width
        self.light = light
        self.camera = camera
        self.objModels = []
        self.shading = 'gouraud'
        self.matrix = np.full(shape=(height, width, 3), fill_value=[0, 0, 0], dtype = np.uint8)
        self.z_buffer = np.full(shape=(height, width), fill_value=np.inf)

    def setShading(self, shading : str):
        if shading != 'gouraud' and shading != 'phong':
            raise ValueError("Данная тонировка не поддерживается")

        self.shading = shading
    
    def getModel(self, i):
        if i >= len(self.objModels):
            raise IndexError("Переданный индекс невалиден")

        return objModels[i]

    def deleteModel(self, i):
        if i >= len(self.objModels):
            raise IndexError("Переданный индекс невалиден")

        del objModels[i]
        
    def addModel(self, scale, objModel : ObjModel):
        self.objModels.append(objModel)

        #self.objModels[len(self.objModels)-1].v[:,0] += ( self.width / 2 ) / scale
        #self.objModels[len(self.objModels)-1].v[:,1] += ( self.height / 2 ) / scale

    def rotateModelEuler(self, degrees, i : int, type='degree'):
        if type=='degree':
            a = {k: radians(v) for k,v in degrees.items()}
        elif type=='radian':
            a = degrees
        else:
            raise ValueError("Неизвестный тип")
            
        X_rotate = np.array([[1, 0, 0], [0, cos(a['x']), -sin(a['x'])], [0, sin(a['x']), cos(a['x'])]])
        Y_rotate = np.array([[cos(a['y']), 0, sin(a['y'])], [0, 1, 0], [-sin(a['y']), 0, cos(a['y'])]])
        Z_rotate = np.array([[cos(a['z']), -sin(a['z']), 0], [sin(a['z']), cos(a['z']), 0], [0, 0, 1]])

        R = X_rotate @ Y_rotate @ Z_rotate

        self.objModels[i].v = (R @ self.objModels[i].v.T).T
        self.objModels[i].fn = (R @ self.objModels[i].fn.T).T
        self.objModels[i].vn = (R @ self.objModels[i].vn.T).T

    def rotateModelQuaternion(self, degree, vector, i : int, type='degree'):
        vector /= np.linalg.norm(vector)

        if type=='degree':
            radian = radians(degree / 2)
        elif type=='radian':
            radian = degree
        else:
            raise ValueError("Неизвестный тип")
            
        a = cos(radian)
        b = vector[0] * sin(radian)
        c = vector[1] * sin(radian)
        d = vector[2] * sin(radian)
        R = np.array([
            [a*a + b*b - c*c - d*d, 2*b*c - 2*a*d, 2*b*d + 2*a*c],
            [2*b*c + 2*a*d, a*a - b*b + c*c - d*d, 2*c*d - 2*a*b],
            [2*b*d - 2*a*c, 2*c*d + 2*a*b, a*a - b*b - c*c + d*d]
        ])

        self.objModels[i].v = (R @ self.objModels[i].v.T).T
        self.objModels[i].fn = (R @ self.objModels[i].fn.T).T
        self.objModels[i].vn = (R @ self.objModels[i].vn.T).T
        

    def scaleModel(self, init_scale, i : int):
        self.objModels[i].v *= init_scale

    def shiftModel(self, model_shift, i : int):
        self.objModels[i].v += model_shift

    def perspectiveTransformModel(self, camera_shift_z, init_scale, i : int):
        z_shift = 1.1*abs(self.objModels[i].v[:, 2].min())

        self.objModels[i].v[:, 2] += z_shift
        self.objModels[i].v[:, 2] /= init_scale
        self.objModels[i].v[:, 2] += camera_shift_z     
        
        self.objModels[i].v[:, 0] = self.objModels[i].v[:, 0] / (self.objModels[i].v[:, 2]) + self.width / 2
        self.objModels[i].v[:, 1] = self.objModels[i].v[:, 1] / (self.objModels[i].v[:, 2]) + self.height / 2
    
    def renderScene(self, filename : str):
        
        for model in self.objModels:
            i = 0
            for face in model.f:
                dividedFaces = []
                numberOfDots = len(face[:, 0])
                for dot in range(2, numberOfDots):
                    newFace = []
                    newFace.append(face[0])
                    newFace.append(face[dot-1])
                    newFace.append(face[dot])
                    dividedFaces.append(np.array(newFace))

                for dividedFace in dividedFaces:
                    norm = model.fn[i]
                    angle = np.dot(norm, self.camera)/np.linalg.norm(norm)
                    if (angle < 0):
                        self.draw_triangle(model, dividedFace)
                i += 1    
        Image.fromarray(np.rot90(self.matrix, 2), 'RGB').save(filename)

    def draw_triangle(self, model, face):
        polygon = face[:, 0]      
        normals = face[:, 2]
        
        y = np.array([model.v[polygon[i]][0] for i in range(len(polygon))])
        x = np.array([model.v[polygon[i]][1] for i in range(len(polygon))])
        z = np.array([model.v[polygon[i]][2] for i in range(len(polygon))])
        
        ymin = max(int(min(y)), 0)
        ymax = min(int(max(y)), self.width - 1)
        xmin = max(int(min(x)), 0)
        xmax = min(int(max(x)), self.height - 1)
    
        uv = np.array([[model.vt[face[j, 1]][i] for j in range(len(polygon))] for i in range(2)])

        if self.shading == 'gouraud':
            I = np.array([min(np.dot(model.vn[polygon[i]], self.light)/(np.linalg.norm(model.vn[polygon[i]]) * np.linalg.norm(self.light)), 0) for i in range(len(polygon))])
    
        for i in range(xmin, xmax + 1):
            for j in range(ymin, ymax + 1):
                coords = self.barycentric_coordinates(i, j, x, y)
                z_coord = np.dot(coords, z)
    
                if z_coord < self.z_buffer[i, j] and (coords >= 0).all():
                    self.z_buffer[i, j] = z_coord
                    if self.shading == 'gouraud':
                        coef = np.dot(coords, I)
                    elif self.shading == 'phong':
                        pixel_normal = model.vn[polygon[0]] * coords[0] + model.vn[polygon[1]] * coords[1] + model.vn[polygon[2]] * coords[2]
                        coef = min(np.dot(pixel_normal, self.light)/(np.linalg.norm(pixel_normal) * np.linalg.norm(self.light)), 0)
                        
                    pixel_index = [round(model.t.shape[1] * np.dot(coords, uv[i])) for i in range(2)]
                    color = model.t[pixel_index[1]][pixel_index[0]]
                    self.matrix[i, j] = color * -coef

    def barycentric_coordinates(self, i, j, x, y):
        system = np.vstack((np.ones(3), x, y))
        column = np.array([1, i, j])
        
        return np.linalg.solve(system, column)