import numpy as np

# Threshold for floating-point zero
ORTH_EPSILON = 1.0e-9

# Axes
XAXIS = (1., 0., 0.)
YAXIS = (0., 1., 0.)
ZAXIS = (0., 0., 1.)


class BasisError(Exception):
    pass


def vector(vec, w=None):
    '''
    A vector with optional w-value
    '''
    vec = np.array(vec).astype(np.float64).reshape((3, 1))
    if w is not None:
        vec = np.vstack((vec, [w]))
    return vec


def vector_matrix(vectors, w=None):
    '''
    A matrix of column vectors
    '''
    return np.hstack((vector(v, w) for v in vectors))


def unit_vector_matrix(vectors, w=None):
    '''
    A matrix of normalized column vectors
    '''
    matrix = np.hstack((vector(v) / np.linalg.norm(v) for v in vectors))
    if w is not None:
        matrix = np.vstack((matrix, w*np.ones(matrix.shape[1])))
    return matrix


def rotation_matrix(point, direction, angle):
    '''
    4x4 matrix to rotate about the line defined by the given point and direction
    '''
    a, b, c = point
    u, v, w = direction
    return np.array([
        [u**2 + (v**2 + w**2)*np.cos(angle), u*v*(1 - np.cos(angle)) - w*np.sin(angle), u*w*(1 - np.cos(angle))
         + v*np.sin(angle), (a*(v**2 + w**2) - u*(b*v + c*w))*(1 - np.cos(angle)) + (b*w - c*v)*np.sin(angle)],
        [u*v*(1 - np.cos(angle)) + w*np.sin(angle), v**2 + (u**2 + w**2)*np.cos(angle), v*w*(1 - np.cos(angle))
         - u*np.sin(angle), (b*(u**2 + w**2) - v*(a*u + c*w))*(1 - np.cos(angle)) + (c*u - a*w)*np.sin(angle)],
        [u*w*(1 - np.cos(angle)) - v*np.sin(angle), v*w*(1 - np.cos(angle)) + u*np.sin(angle), w**2 + (u**2+ v**2)
         *np.cos(angle), (c*(u**2 + v**2) - w*(a*u + b*v))*(1 - np.cos(angle)) + (a*v - b*u)*np.sin(angle)],
        [0, 0, 0, 1]
    ]).astype(np.float64)


def rotate2d(point, angle):
    '''
    Rotate the given point in 2D
    '''
    return point[0]*np.cos(angle) - point[1]*np.sin(angle), point[1]*np.cos(angle) + point[0]*np.sin(angle)


def project2d(point, center, projection):
    '''
    Projection of a 3D point onto a 2D plane
    '''
    return center[0] + int(point[0]/point[2]*projection[0]), center[1] + int(point[1]/point[2]*projection[1])


def translation_matrix(dx, dy, dz):
    '''
    4x4 translation matrix
    '''
    return np.hstack((np.vstack((np.identity(3), np.zeros(3))), vector((dx, dy, dz), w=1)))


def basis_matrix(basis):
    if is_orthogonal_basis(basis[0], basis[1], basis[2]):
        return np.hstack((unit_vector_matrix(basis, w=0), vector((0, 0, 0), w=1)))
    else:
        raise BasisError(f'Vectors are not orthogonal or not right-handed')


def is_orthogonal_basis(x, y, z):
    '''
    Check if the given vectors form a right-handed orthogonal basis. x, y, z must be unit vectors.
    '''
    return np.linalg.norm(np.cross(np.ravel(x), np.ravel(y)) - np.ravel(z)) < ORTH_EPSILON


def rotate_basis(basis_mat, direction, angle):
    '''
    Rotate a basis matrix about the given point and direction
    '''
    return np.matmul(rotation_matrix((0, 0, 0), direction, angle), basis_mat)


def oriented_basis(yaw=0, pitch=0, roll=0):
    '''
    Transform a space to a diferent orientation
    '''
    # Create global basis
    basis_mat = unit_vector_matrix((XAXIS, YAXIS, ZAXIS), w=0)

    # Perform rotations
    basis_mat = rotate_basis(basis_mat, basis_mat[:3, 2], yaw)
    basis_mat = rotate_basis(basis_mat, basis_mat[:3, 0], pitch)
    basis_mat = rotate_basis(basis_mat, basis_mat[:3, 1], roll)

    return [tuple(basis_mat[:3, i]) for i in range(3)]
