import numpy as np
import pygame

from manager import ModelManager
from models import Camera
from linalg import rotate2d, project2d

SCALE_FACTOR = 400  # scaling factor to slow down mouse movements
MOTION_THRESHOLD = 200  # prevent overly erratic mouse movements


class Scene:
    '''
    Manages input/output of the pygame display
    '''
    def __init__(self):
        self._screen_size = (400, 400)
        self._center = (self._screen_size[0]//2, self._screen_size[1]//2)
        self._screen = None
        self._clock = None
        self._background = (128, 128, 255)
        self._title = None

        self._model_manager = None
        self._camera = Camera()

    def add_manager(self, model_manager):
        '''
        Register a ModelManager with the Scene
        Note that this will replace any existing manager
        '''
        if not isinstance(model_manager, ModelManager):
            raise TypeError('Input must be of type ModelManager')
        self._model_manager = model_manager

    def set_viewpoint(self, viewpoint):
        '''
        Set the camera position (viewpoint is a 3-element iterable)
        '''
        self._camera.viewpoint = viewpoint

    def get_viewpoint(self):
        '''
        Get the camera position
        '''
        return self._camera.viewpoint

    def set_rotation(self, rotation):
        '''
        Set the rotation. rotation is a 2-element iterable, with the first element about
        the camera x-axis (up/down) and the second element about the camera y-axis (side to side).
        '''
        self._camera.rotation = rotation

    def get_rotation(self):
        '''
        Get the camera rotation state
        '''
        return self._camera.rotation

    def set_screen_size(self, width, height):
        '''
        Set the screen size. This should be done before Scene.run() is invoked.
        '''
        self._screen_size = (width, height)

    def set_title(self, title):
        '''
        Set the pygame window title. This should be done before Scene.run() is invoked.
        '''
        self._title = title

    def set_clipping_plane(self, distance):
        '''
        Set the close clipping plane. This should be done before Scene.run() is invoked.
        '''
        self._camera.clip_plane = distance

    def set_background(self, colour):
        '''
        Set the window background. This should be done before Scene.run() is invoked.
        '''
        if len(colour) != 3:
            raise TypeError('Colours must be RGB tuples')
        self._background = colour

    def run(self, duration=0):
        '''
        Run the Scene. This will create a pygame window with the models contained in the
        ModelManager, orient the camera, and animate the models based on their respective
        MotionMaps (see ModelManager for details). The duration argument specifies the time
        for which to run the scene. A value of 0 will cause the scene to run indefinitely.
        '''
        self._initialize()
        time = 0.
        while True:
            dt = self._clock.tick()/1000.
            time += dt
            if duration and time > duration:
                self._quit()

            self._update_camera(dt, pygame.key.get_pressed())

            for event in pygame.event.get():
                self._handle_event(event)

            self._screen.fill(self._background)
            self._update(time)
            self._draw_models()
            pygame.display.flip()
            pygame.time.wait(5)

    def _initialize(self):
        '''
        Initialization of the pygame window, camera, and controls
        '''
        if self._model_manager is None:
            raise ValueError('No ModelManager has been associated with this Scene')
        pygame.init()
        self._min_z = 1
        if self._title is not None:
            pygame.display.set_caption(self._title)
        self._screen = pygame.display.set_mode(self._screen_size)
        self._center = (self._screen_size[0]//2, self._screen_size[1]//2)
        self._clock = pygame.time.Clock()
        fov = np.pi/2
        self._camera.proj_x = self._screen_size[0]/2/np.tan(fov/2)/(self._screen_size[0]/self._screen_size[1])
        self._camera.proj_y = self._screen_size[1]/2/np.tan(fov/2)
        pygame.event.get()
        pygame.mouse.get_rel()
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    def _update_camera(self, dt, keys):
        '''
        Update the camera position and orientation
        '''
        s = dt*10
        cam_pos = self._camera.viewpoint
        if keys[pygame.K_LSHIFT]:
            cam_pos[1, 0] += s
        if keys[pygame.K_SPACE]:
            cam_pos[1, 0] -= s

        x, y = s*np.sin(self._camera.rotation[1]), s*np.cos(self._camera.rotation[1])
        if keys[pygame.K_w]:
            cam_pos[0, 0] += x
            cam_pos[2, 0] += y
        if keys[pygame.K_s]:
            cam_pos[0, 0] -= x
            cam_pos[2, 0] -= y
        if keys[pygame.K_a]:
            cam_pos[0, 0] -= y
            cam_pos[2, 0] += x
        if keys[pygame.K_d]:
            cam_pos[0, 0] += y
            cam_pos[2, 0] -= x
        self._camera.viewpoint = cam_pos

    def _draw_models(self):
        '''
        Draw the models contained in the ModelManager. Models are drawn face-by-face, with all
        faces from all models being sorted by depth and drawn in reverse order (farthest first).
        '''
        faces_to_render = []
        colours = []
        depths = []
        for key in self._model_manager.models:
            vertices = [self._convert_coords(v) for v in self._model_manager.get_vertices(key, world=True)]
            faces = self._model_manager.get_faces(key)
            colour = self._model_manager.get_colour(key)
            for f in range(len(faces)):
                face_verts = self._clip([vertices[i] for i in faces[f]], self._camera.clip_plane)

                if len(face_verts) > 2:
                    faces_to_render.append([project2d(v, self._center, (self._camera.proj_x, self._camera.proj_y))
                                            for v in face_verts])
                    colours.append(colour)
                    depths.append(sum(sum(v[i]/len(face_verts) for v in face_verts)**2 for i in range(3)))

        # Sort by depth and draw from back to front
        order = sorted(range(len(faces_to_render)), key=lambda i: depths[i], reverse=True)
        for i in order:
            pygame.draw.polygon(self._screen, colours[i], faces_to_render[i])
            for j in range(len(faces_to_render[i])):
                pygame.draw.line(self._screen, (0, 0, 0), faces_to_render[i][j-1], faces_to_render[i][j])

    def _convert_coords(self, vertex):
        '''
        Convert world coordinates to camera coordinates
        '''
        x = vertex[0] - self._camera.viewpoint[0]
        y = vertex[1] - self._camera.viewpoint[1]
        z = vertex[2] - self._camera.viewpoint[2]
        x, z = rotate2d((x, z), self._camera.rotation[1])
        y, z = rotate2d((y, z), self._camera.rotation[0])
        return x, y, z

    def _handle_event(self, event):
        '''
        Handler for mouse/keyboard events
        '''
        if event.type == pygame.MOUSEMOTION:
            self._mouse_motion(event)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._quit()

    def _mouse_motion(self, event):
        '''
        Handler for mouse movement
        '''
        x, y = event.rel
        if abs(x) > MOTION_THRESHOLD or abs(y) > MOTION_THRESHOLD:
            return
        x /= SCALE_FACTOR
        y /= SCALE_FACTOR
        self._camera.rotation = (self.get_rotation()[0] + y, self.get_rotation()[1] + x)

    @staticmethod
    def _quit():
        '''
        Quit the Scene
        '''
        pygame.quit()
        exit()

    def _update(self, time):
        '''
        Update the models in the Scene
        '''
        self._model_manager.update_models(time)

    def _clip(self, face_vertices, clip_dist):
        '''
        Clip parts of the face that are 'behind' the camera (see set_clipping_plane)
        '''
        i = 0
        while i < len(face_vertices):
            if face_vertices[i][2] < clip_dist:
                sides = []
                prev_vert = face_vertices[i - 1]
                next_vert = face_vertices[(i + 1)%len(face_vertices)]
                if prev_vert[2] >= clip_dist:
                    sides.append(self._get_z(face_vertices[i], prev_vert, clip_dist))
                if next_vert[2] >= clip_dist:
                    sides.append(self._get_z(face_vertices[i], next_vert, clip_dist))
                face_vertices = face_vertices[:i] + sides + face_vertices[i + 1:]
                i += len(sides) - 1
            i += 1
        return face_vertices

    @staticmethod
    def _get_z(v1, v2, min_z):
        '''
        Modify vertex for clipped face
        '''
        if v2[2] == v1[2] or min_z < v1[2] or min_z > v2[2]:
            return None
        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]
        dz = v2[2] - v1[2]
        i = (min_z - v1[2])/dz
        return v1[0] + dx*i, v1[1] + dy*i, min_z

