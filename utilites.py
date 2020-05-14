import numpy as np

def direct_linear_transformation(old_points, new_points):
    """Direct Linear Transformation algorithm

    Algorithm for calculating homography matrix based on the "Multiple View
    Geometry in Computer Vision"

    Note:
        Indices of matching points should be the same in both arrays.
        It is assumed that w for each point is equal 1

    Args:
        old_points (np.array):
        new_points (np.array):
    Returns:
        np.array: Homography matrix of shape (3,3)
    Todo:
        Add normalization
    """
    assert len(old_points) == len(new_points)
    n = len(old_points)
    A = np.zeros((2*n, 9))
    for i in range(n):
        x1 = old_points[i][0]
        y1 = old_points[i][1]
        x2 = new_points[i][0]
        y2 = new_points[i][1]

        A[i*2:(i+1)*2, :] = [
            [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2],
            [x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2]
        ]

    U, S, Vh = np.linalg.svd(A)
    h = Vh[-1,:]
    H = h.reshape(3,3)

    return H

def eight_point_algorithm(old_points, new_points):
    """Eight point algorithm

    Algorithm for calculating fundamental matrix based on the "Multiple View
    Geometry in Computer Vision"

    Note:
        Indices of matching points should be the same in both arrays.

    Args:
        old_points (np.array):
        new_points (np.array):
    Returns:
        np.array: Fundamental matrix
    Todo:
        Add normalization
    """
    assert len(old_points) == len(new_points)
    n = len(old_points)
    A = np.zeros((n, 9))
    # TODO: vectorize instead of interating
    for i in range(n):
        x1 = old_points[i][0]
        y1 = old_points[i][1]
        x2 = new_points[i][0]
        y2 = new_points[i][1]
        A[i, :] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    # Linear solution
    U, S, Vh = np.linalg.svd(A)
    f = Vh[-1,:]
    F = f.reshape(3,3)

    # Constraing enforcment

    U, S, Vh = np.linalg.svd(F)
    F_prim = np.matmul(np.matmul(U, np.diag([S[0], S[1], 0])), Vh)

    return F_prim
