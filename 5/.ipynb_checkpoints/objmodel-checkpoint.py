import numpy as np

class ObjModel:
    def __init__(self, v, vt, vn, f):
        self.v = v
        self.vt = vt
        #self.vn = vn
        self.f = f
        self.fn = self.find_all_normals()
        self.vn = self.interpolate_vertex_normals()
        self.t = None

    def addTexture(self, t):
        self.t = t

    def find_all_normals(self):
        fn = []
        for face in self.f:
            polygon = face[:, 0]
            vec1 = self.v[polygon[1]] - self.v[polygon[0]]
            vec2 = self.v[polygon[2]] - self.v[polygon[0]]
            fn.append(np.cross(vec1, vec2))
    
        return np.array(fn)
        
    def find_all_faces_for_vertex(self):
        vertices_to_faces = {}
    
        i = 0
        for face in self.f:
            triangle = face[:, 0]
            for vertex in triangle:
                vertices_to_faces.setdefault(vertex, []).append(i)
            i += 1
        return vertices_to_faces  

    def interpolate_vertex_normals(self):
        vertices_in_faces = self.find_all_faces_for_vertex()
        vertices_normal = {}
        for vertex, faces in vertices_in_faces.items():
            vector_sum = np.zeros(shape=3,)
            for i in faces:
                vector_sum += self.fn[i]
            vertices_normal[vertex] = vector_sum/np.linalg.norm(vector_sum)
        return np.array([value for value in vertices_normal.values()])