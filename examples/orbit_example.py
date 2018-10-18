import numpy as np

from scene import Scene
from manager import ModelManager

BOX = {
    'vertices': [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1)
    ],
    'faces': [
        (0, 2, 6, 4),
        (1, 3, 7, 5),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 1, 3, 2),
        (4, 5, 7, 6),
    ]
}


def xz_circular_motion(time, radius, frequency, y=0):
    ang_freq = 2*np.pi*frequency
    x = radius*np.cos(time*ang_freq)
    z = radius*np.sin(time*ang_freq)
    return x, y, z


def sphere2_position(time):
    '''
    Make sphere2 orbit the origin
    '''
    return xz_circular_motion(time, 15, 0.1)


def sphere3_position(time):
    '''
    Make sphere3 orbit sphere2
    '''
    xc, yc, zc = sphere2_position(time)
    xr, yr, zr = xz_circular_motion(time, 3, 1)
    return xc + xr, yc + yr, zc - zr


def sphere4_position(time):
    '''
    Make sphere4 orbit the origin
    '''
    x, y, z = xz_circular_motion(time, 35, 0.05)
    return x, y, -z


def sphere5_position(time):
    '''
    Make sphere5 orbit sphere4
    '''
    xc, yc, zc = sphere4_position(time)
    xr, yr, zr = xz_circular_motion(time, 10, 0.15)
    return xc + xr, yc + yr, zc + zr


def main():
    # Instantiate manager
    manager = ModelManager()

    # Create box models
    manager.add_model('sphere1', vertices=BOX['vertices'], faces=BOX['faces'])
    manager.add_model('sphere2', vertices=BOX['vertices'], faces=BOX['faces'])
    manager.add_model('sphere3', vertices=BOX['vertices'], faces=BOX['faces'])
    manager.add_model('sphere4', vertices=BOX['vertices'], faces=BOX['faces'])
    manager.add_model('sphere5', vertices=BOX['vertices'], faces=BOX['faces'])

    # Recolour boxes
    manager.set_colour('sphere1', (255, 255, 255))
    manager.set_colour('sphere2', (0, 255, 0))
    manager.set_colour('sphere3', (255, 0, 0))
    manager.set_colour('sphere4', (0, 0, 255))
    manager.set_colour('sphere5', (255, 255, 0))

    # Scale the spheres
    manager.scale('sphere1', 10)
    manager.scale('sphere3', 0.5)
    manager.scale('sphere4', 2)

    # Add motion to the models (box1 is stationary)
    manager.add_motion('sphere2', positions=sphere2_position)
    manager.add_motion('sphere3', positions=sphere3_position)
    manager.add_motion('sphere4', positions=sphere4_position)
    manager.add_motion('sphere5', positions=sphere5_position)

    # Create scene
    scene = Scene()
    scene.set_screen_size(800, 800)
    scene.set_viewpoint((0, -30, -30))
    scene.set_rotation((np.pi/3, 0))
    scene.set_title('Orbit Example')

    # Add manager to scene
    scene.add_manager(manager)

    # Run scene
    scene.run(duration=60)


if __name__ == '__main__':
    main()
