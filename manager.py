from models import Model, MotionMap
from linalg import vector, oriented_basis


class ModelManager:
    '''
    Controls a set of models
    '''
    def __init__(self):
        self._models = {}
        self._motions = {}

    def add_model(self, key, **kwargs):
        '''
        Register a new model. Models may be passed in three ways via keyword args:
            model:              pass in a Model object
            stl_file:           pass in an STL file to create the model from
            vertices, faces:    pass in a list of vertices (required) and a list of faces (optional)
        '''
        if key in self._models:
            raise KeyError(f'A model already exists with key {key}')
        model = kwargs.get('model', None)
        if model is not None:
            if not isinstance(model, Model):
                raise TypeError(f'Model of type {type(model).__name__} cannot be added')
        else:
            stl_file = kwargs.get('stl_file', None)
            if stl_file is not None:
                model = Model.from_stl(stl_file)
            else:
                vertices = kwargs.get('vertices', None)
                if vertices is not None:
                    model = Model(vertices=vertices)
                    faces = kwargs.get('faces', None)
                    if faces is not None:
                        model.faces = faces
        if model is None:
            raise ValueError('Could not construct a model from the given inputs')
        self._models[key] = model

    def remove_model(self, key):
        '''
        Remove a model from the manager
        '''
        del self._models[key]
        if key in self._motions:
            del self._motions[key]

    def add_motion(self, key, positions=(), orientations=(), times=()):
        '''
        Add motion to a model. Motion can be specified as:
            a) List of positions, orientations, and times
                In this case, motion is defined as a discrete set of states. It is assumed that
                nothing changes between timepoints, as the discrete states are converted to a
                piecewise continuous function behind the scenes. Positions and orientations must
                both be passed as nx3 iterables - for positions, the 3 is x-y-z, and for orientations,
                the 3 is yaw-pitch-roll. Note that rotations are done in the order yaw-pitch-roll about
                the z, x, and y axes of the object respectively, with resepct to the world basis.
            b) Functional relationships
                In this case, positions and orientations are passed as functions of time. The functions
                must only take a single argument (time), and must return a 3-element iterable in both cases.
                See case a) for a description of the 3 elements.
        Note that in both cases, positions and orientations can be passed independently (i.e.
        one may pass in an argument for positions, but not orientations). Also note that adding motion
        implies replacing any existing motion, as motions do not stack.
        '''
        self._motions[key] = MotionMap(positions, orientations, times)

    def remove_motion(self, key):
        '''
        Remove motion from an object
        '''
        del self._motions[key]

    def translate(self, key, translation):
        '''
        Translate a model in the world coordinate system
        '''
        self._models[key].origin = self._models[key].origin + vector(translation)

    def orient(self, key, yaw, pitch, roll):
        '''
        Orient a model with respect to the global basis
        '''
        basis = oriented_basis(yaw, pitch, roll)
        self._models[key].basis = basis

    def scale(self, key, factor):
        '''
        Scale a model. Note that this scales the model while maintaining
        its position and orientation, and that the world scale is unchanged.
        '''
        self._models[key].scale(factor)

    def change_local_basis(self, key, basis):
        '''
        Change the local basis of a model. By default, models are defined with an
        intrinsic coordinate system (the model space). It is possible to modify this
        intrinsic space such that an orientation in the world space produces the desired
        orientation of the object. For example, if an object's vertices are defined upside-
        down (y flipped), one could change the local basis to correct this.
        '''
        self._models[key].change_local_basis(basis)

    # may be added later - requires some work
    # def rotate(self, key, point, direction, angle):
    #     rotated_space = rotate_space(self._models[key].space, point, direction, angle)
    #     self._models[key].set_space(rotated_space)

    def get_basis(self, key):
        '''
        Get the basis vectors of a model
        '''
        return self._models[key].basis

    def set_basis(self, key, basis):
        '''
        Set the basis vectors of a model. basis must be an iterable containing 3 vectors.
        '''
        self._models[key].basis = basis

    def get_position(self, key):
        '''
        Get the position of a model
        '''
        return self._models[key].origin

    def set_position(self, key, position):
        '''
        Set the position of a model. position must be a 3-element iterable.
        '''
        self._models[key].origin = position

    def get_vertices(self, key, world=False):
        '''
        Get a list of a model's vertices
        '''
        if world:
            return self._models[key].world_vertices
        return self._models[key].vertices

    def get_colour(self, key):
        '''
        Get the current colour of a model
        '''
        return self._models[key].colour

    def set_colour(self, key, colour):
        '''
        Set the colour of a model. colour must be an RGB triplet.
        '''
        self._models[key].colour = colour

    def get_faces(self, key):
        '''
        Return a list of faces of a model. Note that faces are described by the
        indices of their vertices, and so this list is meaningless without a
        corresponding list of vertices (see get_vertices).
        '''
        return self._models[key].faces

    def update_models(self, time):
        '''
        Update the position and orientation of all models based on their MotionMaps
        '''
        states = self._get_states(time)
        for key in states:
            if states[key][0] is not None:
                self.set_position(key, states[key][0])
            if states[key][1] is not None:
                self.orient(key, states[key][1][0], states[key][1][1], states[key][1][2])

    @property
    def models(self):
        '''
        List of the current registered models
        '''
        return [model for model in self._models]

    def _get_states(self, time):
        '''
        Get the current state of each model from its MotionMap
        '''
        states = {}
        for key in self._motions:
            states[key] = self._motions[key].get_state(time)
        return states
