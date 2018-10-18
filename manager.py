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
        del self._models[key]
        if key in self._motions:
            del self._motions[key]

    def add_motion(self, key, positions=(), orientations=(), times=()):
        self._motions[key] = MotionMap(positions, orientations, times)

    def remove_motion(self, key):
        del self._motions[key]

    def translate(self, key, translation):
        self._models[key].origin = self._models[key].origin + vector(translation)

    def orient(self, key, yaw, pitch, roll):
        basis = oriented_basis(yaw, pitch, roll)
        self._models[key].basis = basis

    def scale(self, key, factor):
        self._models[key].scale(factor)

    def change_local_basis(self, key, basis):
        self._models[key].change_local_basis(basis)

    # may be added later - requires some work
    # def rotate(self, key, point, direction, angle):
    #     rotated_space = rotate_space(self._models[key].space, point, direction, angle)
    #     self._models[key].set_space(rotated_space)

    def get_basis(self, key):
        return self._models[key].basis

    def set_basis(self, key, basis):
        self._models[key].basis = basis

    def get_position(self, key):
        return self._models[key].origin

    def set_position(self, key, position):
        self._models[key].origin = position

    def get_vertices(self, key, world=False):
        if world:
            return self._models[key].world_vertices
        return self._models[key].vertices

    def get_colour(self, key):
        return self._models[key].colour

    def set_colour(self, key, colour):
        self._models[key].colour = colour

    def get_edges(self, key):
        return self._models[key].edges

    def get_faces(self, key):
        return self._models[key].faces

    def update_models(self, time):
        states = self._get_states(time)
        for key in states:
            if states[key][0] is not None:
                self.set_position(key, states[key][0])
            if states[key][1] is not None:
                self.orient(key, states[key][1][0], states[key][1][1], states[key][1][2])

    @property
    def models(self):
        return [model for model in self._models]

    def _get_states(self, time):
        states = {}
        for key in self._motions:
            states[key] = self._motions[key].get_state(time)
        return states
