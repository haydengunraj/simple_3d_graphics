import stl
import numpy as np

from linalg import vector, vector_matrix, basis_matrix, translation_matrix, XAXIS, YAXIS, ZAXIS


class Space:
    '''
    Representation of a space in world coordinates
    '''
    def __init__(self, basis=(XAXIS, YAXIS, ZAXIS), origin=(0, 0, 0)):
        '''
        basis:  a list of the basis vectors of the space
        origin: the position of the center of the space
        '''
        self._basis = None
        self._translation = None
        self._inverse = None

        if basis is not None:
            self.basis = basis
        if origin is not None:
            self.origin = origin

    def invert(self, points):
        '''
        Convert a point in the Space's coordinates to world coordinates
        '''
        return np.matmul(self._inverse, points)

    @property
    def origin(self):
        '''
        Origin getter
        '''
        return self._translation[:3, 3]

    @origin.setter
    def origin(self, origin):
        '''
        Origin setter
        '''
        self._translation = translation_matrix(origin[0], origin[1], origin[2])
        self._update_inverse()

    @property
    def basis(self):
        '''
        Basis getter
        '''
        return self._basis[:3, :3]

    @basis.setter
    def basis(self, basis):
        '''
        Basis setter
        '''
        self._basis = basis_matrix(basis)
        self._update_inverse()

    def _update_inverse(self):
        '''
        Update the inverse transformation matrix
        '''
        if self._basis is not None and self._translation is not None:
            self._inverse = np.matmul(self._translation, np.linalg.inv(self._basis))


class Model:
    '''
    Simple wireframe representation of an object
    '''
    def __init__(self, vertices=None, faces=None, colour=(255, 0, 0)):
        '''
        vertices:   a list of the model's vertices
        faces:      a list of the model's faces, described by the indices of their vertices
        colour:     an RGB triplet describing the model's colour
        '''
        self._vertices = np.zeros((4, 0))
        self._faces = []
        self._colour = colour
        self._space = Space()  # initial Space is the same as the world frame

        if vertices is not None:
            self.vertices = vertices
            if faces is not None:
                self.faces = faces

        self.set_local_center(self.centroid)

    @property
    def origin(self):
        '''
        Position of the model in world coordinates
        '''
        return self._space.origin.reshape(3, 1)

    @origin.setter
    def origin(self, origin):
        '''
        Set the origin of the model space in world coordinates
        '''
        self._space.origin = origin

    @property
    def basis(self):
        '''
        Basis vectors of the model in world coordinates
        '''
        return [tuple(self._space.basis[:, i] for i in range(3))]

    @basis.setter
    def basis(self, basis):
        '''
        Set the basis vectors of the model in world coordinates
        '''
        self._space.basis = basis

    @property
    def vertices(self):
        '''
        List of vertices of the model
        '''
        return [tuple(self._vertices[:3, i]) for i in range(self._vertices.shape[1])]

    @vertices.setter
    def vertices(self, vertices):
        '''
        Set the model vertices
        '''
        self._vertices = vector_matrix(vertices, w=1)

    @property
    def faces(self):
        '''
        List of faces of the model, specified by their vertex indices
        '''
        return self._faces

    @faces.setter
    def faces(self, faces):
        '''
        Set the faces of the model. Note that this erases any existing faces.
        To add new faces, use add_faces instead.
        '''
        self._faces = []
        self.add_faces(faces)

    def add_faces(self, faces):
        '''
        Add new faces to the model
        '''
        for f in faces:
            self._add_face(f)

    @property
    def colour(self):
        '''
        Current model colour
        '''
        return self._colour

    @colour.setter
    def colour(self, colour):
        '''
        Set model colour
        '''
        if len(colour) != 3:
            raise TypeError('Colours must be RGB triplets')
        self._colour = tuple(colour)

    @property
    def centroid(self):
        '''
        Centroid of the model in world coordinates
        '''
        return np.sum(self._vertices[:3], axis=1)/self._vertices.shape[1]

    @property
    def world_vertices(self):
        '''
        List of model vertices in world coordinates
        '''
        world_vertices = self._space.invert(self._vertices)
        return [tuple(world_vertices[:3, i]) for i in range(self._vertices.shape[1])]

    def set_local_center(self, center):
        '''
        Set the local center of the model. This is the point at which
        the origin is assumed to be.
        '''
        tmat = translation_matrix(-center[0], -center[1], -center[2])
        self._vertices = np.matmul(tmat, self._vertices)

    def change_local_basis(self, basis):
        '''
        Change the local basis of the model. This allows the model to be oriented
        locally, without technically changing the orientation in the world space.
        '''
        current_basis = [tuple(self._space.basis[:, i]) for i in range(3)]
        self._space.basis = basis
        self._vertices = self._space.invert(self._vertices)
        self._space.basis = current_basis

    def scale(self, factor):
        '''
        Scale the model about its origin
        '''
        self._vertices[:3, :] = self._vertices[:3, :]*factor

    @classmethod
    def from_stl(cls, stl_file):
        '''
        Create a model from an STL file
        '''
        vertices, faces = cls._convert_stl(stl_file)
        return cls(vertices=vertices, faces=faces)

    def _add_face(self, face):
        '''
        Add a face to the model
        '''
        if any((i < 0 or i >= self._vertices.shape[1] for i in face)):
            raise IndexError('Face contains vertices which do not exist')
        self._faces.append(tuple(i for i in face))

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
        '''
        positions:      iterable or callable defining position over time
        orientations:   iterable or callable defining position over time
        times:          iterable of times

        Note: if using non-empty iterables, the iterables must have the same length
        and are assumed to be parallel (i.e. at time times[0], position is positions[0])
        '''
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
        '''
        Get the state of motion at the given time
        '''
        return self._position(time), self._orientation(time)

    @staticmethod
    def _piecewise(times, vals):
        '''
        Turns a list of times and vals into a continuous function, v(t)
        '''
        times = np.array(times[:len(vals)]).ravel()
        vals = list(vals)
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
        '''
        Position of the camera in world coordinates
        '''
        return self._viewpoint

    @viewpoint.setter
    def viewpoint(self, viewpoint):
        '''
        Set the camera position in world coordinates
        '''
        self._viewpoint = vector(viewpoint)

    @property
    def rotation(self):
        '''
        Rotation state of the camera
        '''
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        '''
        Set the rotation state of the camera. Must be a 2-element iterable.
        '''
        rotation = tuple(rotation)
        if len(rotation) != 2:
            raise TypeError('Rotation must have length 2')
        self._rotation = rotation
