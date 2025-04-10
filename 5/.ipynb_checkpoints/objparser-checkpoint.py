import numpy as np
from PIL import Image
from PIL import ImageOps
from objmodel import ObjModel

class ObjParser:

    @staticmethod
    def processTexture(filename: str):
        with Image.open(filename) as uv:
            img = uv.convert('RGB')
        img = ImageOps.flip(img)
        return np.array(img)

    @staticmethod
    def processObj(filename : str):    
        with open(filename) as obj:
            obj.seek(0, 2)
            length = obj.tell()
            obj.seek(0, 0)
    
            vertices = []
            vertices_texture = []
            vertices_normal = []
            faces = []
            
            while obj.tell() != length:
                line = obj.readline().split()
    
                if len(line) == 0:
                    continue
                
                if line[0] == 'v':
                    vertices.append(list(map(float, line[1:])))
                elif line[0] == 'vt':
                    vertices_texture.append(list(map(float, line[1:])))
                elif line[0] == 'vn':
                    vertices_normal.append(list(map(float, line[1:])))
                elif line[0] == 'f':
                    faces.append([[int(num) - 1 for num in vertex.split('/')] for vertex in line[1:]])
            return ObjModel(np.array(vertices), np.array(vertices_texture), np.array(vertices_normal), np.array(faces))