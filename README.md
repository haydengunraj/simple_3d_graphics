## Simple 3D Graphics

In an attempt to get a better understanding of how 3D graphics work, this project implements a (very) simple graphics engine using Python 3 with Numpy and Pygame. While it isn't suitable for anything even remotely intensive, it's a neat little tool for manipulating simple objects. While I may be making some basic changes in the future, I have no real plans to continue development. However, I will likely be looking into creating a similar set of tools using OpenGL.

![Orbit](https://github.com/haydengunraj/simple_3d_graphics/blob/master/examples/data/orbit.gif "Orbit")

### Usage

The examples directory contains a couple of simple examples to show how the code works. Put simply, the features include:

- Creation of objects from vertices/faces, or from .stl files
- Animation of objects from discrete position/orientations or functions describing the motion
- Navigation of the scene using Minecraft-esque controls (w, a, s, d, shift, space)
- Manipulation of model and world spaces

### Requirements

- [Python 3 (I used 3.7)](https://www.python.org/)
- [Numpy](http://www.numpy.org/)
- [Pygame](https://www.pygame.org/)
- [numpy-stl](https://pypi.org/project/numpy-stl/)