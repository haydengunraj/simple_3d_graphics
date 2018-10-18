import numpy as np
import csv
from datetime import datetime
from pathlib import Path

from scene import Scene
from manager import ModelManager


def get_data(filename, headersize=0):
    with open(filename, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        n_cols = len(next(reader))
        cols = [[] for _ in range(n_cols)]
        csv_file.seek(0)
        for _ in range(headersize):
            next(reader)
        for row in reader:
            for c in range(n_cols):
                cols[c].append(row[c])
    return cols


def main():
    # Instantiate manager
    manager = ModelManager()

    # Load paddle model from STL
    data_dir = Path(__file__).absolute().parent.joinpath('data')
    manager.add_model('paddle', stl_file=data_dir.joinpath('paddle.stl'))

    # Re-define local coordinate system of paddle
    manager.change_local_basis('paddle', ((1, 0, 0), (0, 0, -1), (0, 1, 0)))

    # Get motion data from a csv
    data = get_data(data_dir.joinpath('paddle.csv'), headersize=1)

    # Convert to correct data types
    times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f') for t in data[0]]
    times = [(t - times[0]).total_seconds() for t in times]
    yaw = [float(v) for v in data[1]]
    pitch = [float(v) for v in data[2]]
    roll = [float(v) for v in data[3]]
    orientations = [(yaw[i]/180*np.pi, -pitch[i]/180*np.pi, -roll[i]/180*np.pi) for i in range(len(yaw))]

    # Add motion to the model
    manager.add_motion('paddle', times=times, orientations=orientations)

    # Create scene
    scene = Scene()
    scene.set_screen_size(800, 800)
    scene.set_viewpoint((0, 0, -5))
    scene.set_title('Paddle Example')

    # Add manager to scene
    scene.add_manager(manager)

    # Run scene
    scene.run(duration=60)


if __name__ == '__main__':
    main()
