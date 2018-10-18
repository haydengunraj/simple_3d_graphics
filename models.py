import stl
import numpy as np

from linalg import vector, vector_matrix, basis_matrix, translation_matrix, XAXIS, YAXIS, ZAXIS


class Space:
    def __init__(self, basis=(XAXIS, YAXIS, ZAXIS), origin=(0, 0, 0)):
        self._basis = None
        self._translation = None
        self._inverse = None
        self.update(basis=basis, origin=origin)

    def update(self, basis=None, origin=None):
        if basis is not None:
            self._basis = basis_matrix(basis)
        if origin is not None:
            self._translation = translation_matrix(origin[0], origin[1], origin[2])
        self._inverse = np.matmul(self._translation, np.linalg.inv(self._basis))

    def invert(self, points):
        return np.matmul(self._inverse, points)

    @property
    def origin(self):
        return self._translation[:3, 3]

    @origin.setter
    def origin(self, origin):
        self._translation = translation_matrix(origin[0], origin[1], origin[2])
        self._inverse = np.matmul(self._translation, np.linalg.inv(self._basis))

    @property
    def basis(self):
        return self._basis[:3, :3]

    @basis.setter
    def basis(self, basis):
        self._basis = basis_matrix(basis)
        self._inverse = np.matmul(self._translation, np.linalg.inv(self._basis))


class Model:
    '''
    Simple wireframe representation of an object
    '''
    def __init__(self, vertices=None, faces=None, colour=(255, 0, 0)):
        self._vertices = np.zeros((4, 0))
        self._faces = []
        self._colour = colour
        self._space = Space()

        if vertices is not None:
            self.vertices = vertices
            if faces is not None:
                self.faces = faces

        self.set_local_center(self.centroid)

    @property
    def origin(self):
        return self._space.origin.reshape(3, 1)

    @origin.setter
    def origin(self, origin):
        self._space.update(origin=origin)

    @property
    def basis(self):
        return [tuple(self._space.basis[:, i] for i in range(3))]

    @basis.setter
    def basis(self, basis):
        self._space.update(basis=basis)

    @property
    def vertices(self):
        return [tuple(self._vertices[:3, i]) for i in range(self._vertices.shape[1])]

    @vertices.setter
    def vertices(self, vertices):
        self._vertices = vector_matrix(vertices, w=1)

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, faces):
        for f in faces:
            self._add_face(f)

    @property
    def colour(self):
        return self._colour

    @colour.setter
    def colour(self, colour):
        if len(colour) != 3:
            raise TypeError('Colours must be RGB triplets')
        self._colour = tuple(colour)

    @property
    def centroid(self):
        return np.sum(self._vertices[:3], axis=1)/self._vertices.shape[1]

    @property
    def world_vertices(self):
        world_vertices = self._space.invert(self._vertices)
        return [tuple(world_vertices[:3, i]) for i in range(self._vertices.shape[1])]

    def set_local_center(self, center):
        tmat = translation_matrix(-center[0], -center[1], -center[2])
        self._vertices = np.matmul(tmat, self._vertices)

    def change_local_basis(self, basis):
        current_basis = [tuple(self._space.basis[:, i]) for i in range(3)]
        self._space.update(basis=basis)
        self._vertices = self._space.invert(self._vertices)
        self._space.update(basis=current_basis)

    def scale(self, factor):
        self._vertices[:3, :] = self._vertices[:3, :]*factor

    def _add_face(self, face):
        if any((i < 0 or i >= self._vertices.shape[1] for i in face)):
            raise IndexError('Face contains vertices which do not exist')
        self._faces.append(tuple(i for i in face))

    @classmethod
    def from_stl(cls, stl_file):
        vertices, faces = cls._convert_stl(stl_file)
        return cls(vertices=vertices, faces=faces)

    @staticmethod
    def _convert_stl(file):
        '''
        Convert an STL to a set of vertices and faces
        '''
        mesh = stl.mesh.Mesh.from_file(file)
        vertices, inv = np.unique(np.vstack(mesh.vectors), axis=0, return_inverse=True)
        vertices = [tuple(vertex) for vertex in vertices]
        vert_idx = np.array(range(len(vertices)))

        faces = []
        for i in range(mesh.vectors.shape[0]):
            faces.append(tuple(vert_idx[inv[[3*i, 3*i + 1, 3*i + 2]]]))
        return vertices, faces


class MotionMap:
    '''
    Defines motion over time
    '''
    def __init__(self, positions=(), orientations=(), times=()):
        self._position = None
        self._orientation = None

        if callable(positions):
            self._position = positions
        else:
            self._position = self._piecewise(times, positions)

        if callable(orientations):
            self._orientation = orientations
        else:
            self._orientation = self._piecewise(times, orientations)

    def get_state(self, time):
        return self._position(time), self._orientation(time)

    @staticmethod
    def _piecewise(times, vals):
        '''
        Turns a list of times and vals into a continuous function, v(t)
        '''
        times = np.array(times[:len(vals)]).ravel()
        last_idx = [0]  # list so that it is mutable

        def v(t):
            out = None
            indices = np.where(times[last_idx[0]:] > t)[0]
            if len(indices):
                idx = indices[0] + last_idx[0]
                if idx:
                    idx -= 1
                    out = vals[idx]
                    last_idx[0] = idx
            return out
        return v


class Camera:
    '''
    Manages the camera
    '''
    def __init__(self, clip_plane=1, viewpoint=(0, 0, 0), rotation=(0, 0)):
        self._viewpoint = viewpoint
        self._rotation = rotation
        self.clip_plane = clip_plane
        self.proj_x = None
        self.proj_y = None

    @property
    def viewpoint(self):
        return self._viewpoint

    @viewpoint.setter
    def viewpoint(self, viewpoint):
        self._viewpoint = vector(viewpoint)

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        if len(rotation) != 2:
            raise TypeError('Rotation must have length 2')
        self._rotation = tuple(rotation)
