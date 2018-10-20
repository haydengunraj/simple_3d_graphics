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
    '''
    Circular motion in the xz plane
    '''
    ang_freq = 2*np.pi*frequency
    x = radius*np.cos(time*ang_freq)
    z = radius*np.sin(time*ang_freq)
    return x, y, z


def cube2_position(time):
    '''
    Make cube2 orbit the origin
    '''
    return xz_circular_motion(time, 15, 0.1)


def cube3_position(time):
    '''
    Make cube3 orbit cube2
    '''
    xc, yc, zc = cube2_position(time)
    xr, yr, zr = xz_circular_motion(time, 3, 1)
    return xc + xr, yc + yr, zc - zr


def cube4_position(time):
    '''
    Make cube4 orbit the origin
    '''
    x, y, z = xz_circular_motion(time, 35, 0.05)
    return x, y, -z


def cube5_position(time):
    '''
    Make cube5 orbit cube4
    '''
    xc, yc, zc = cube4_position(time)
    xr, yr, zr = xz_circular_motion(time, 10, 0.15)
    return xc + xr, yc + yr, zc + zr


def main():
    # Instantiate manager
    manager = ModelManager()

    # Create box models
    manager.add_model('cube1', vertices=BOX['vertices'], faces=BOX['faces'])
    manager.add_model('cube2', vertices=BOX['vertices'], faces=BOX['faces'])
    manager.add_model('cube3', vertices=BOX['vertices'], faces=BOX['faces'])
    manager.add_model('cube4', vertices=BOX['vertices'], faces=BOX['faces'])
    manager.add_model('cube5', vertices=BOX['vertices'], faces=BOX['faces'])

    # Recolour cubes
    manager.set_colour('cube1', (255, 255, 255))
    manager.set_colour('cube2', (0, 255, 0))
    manager.set_colour('cube3', (255, 0, 0))
    manager.set_colour('cube4', (0, 0, 255))
    manager.set_colour('cube5', (255, 255, 0))

    # Scale the cubes
    manager.scale('cube1', 10)
    manager.scale('cube3', 0.5)
    manager.scale('cube4', 2)

    # Add motion to the models (box1 is stationary)
    manager.add_motion('cube2', positions=cube2_position)
    manager.add_motion('cube3', positions=cube3_position)
    manager.add_motion('cube4', positions=cube4_position)
    manager.add_motion('cube5', positions=cube5_position)

    # Create scene
    scene = Scene()
    scene.set_screen_size(800, 800)
    scene.set_viewpoint((0, -30, -30))
    scene.set_rotation((np.pi/3, 0))
    scene.set_title('Orbit Example')

    # Add manager to scene
    scene.add_manager(manager)

    # Run scene
    scene.run()


if __name__ == '__main__':
    main()
