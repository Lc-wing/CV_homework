import numpy as np
import utils


def find_projection(pts2d, pts3d):
    """
    Computes camera projection matrix M that goes from world 3D coordinates
    to 2D image coordinates.

    [u v 1]^T === M [x y z 1]^T

    Where (u,v) are the 2D image coordinates and (x,y,z) are the world 3D
    coordinates

    Inputs:
    - pts2d: Numpy array of shape (N,2) giving 2D image coordinates
    - pts3d: Numpy array of shape (N,3) giving 3D world coordinates

    Returns:
    - M: Numpy array of shape (3,4)

    """
    # p x (M X) = 0 gives two independent linear constraints per point.
    # With X_h = [x, y, z, 1], the rows are [0, X_h, -v X_h] and [X_h, 0, -u X_h].
    pts2d = np.asarray(pts2d, dtype=float)
    pts3d = np.asarray(pts3d, dtype=float)
    n = pts2d.shape[0]
    X = np.hstack([pts3d, np.ones((n, 1))])
    u = pts2d[:, 0]
    v = pts2d[:, 1]
    A = np.zeros((2 * n, 12))
    A[0::2, 4:8] = X
    A[0::2, 8:12] = -v[:, None] * X
    A[1::2, 0:4] = X
    A[1::2, 8:12] = -u[:, None] * X
    # Unit singular vector for the smallest singular value: A m = 0.
    _, _, vt = np.linalg.svd(A)
    M = vt[-1].reshape(3, 4)
    return M

def compute_distance(pts2d, pts3d):
    """
    use find_projection to find matrix M, then use M to compute the average 
    distance in the image plane (i.e., pixel locations) 
    between the homogeneous points M X_i and 2D image coordinates p_i

    Inputs:
    - pts2d: Numpy array of shape (N,2) giving 2D image coordinates
    - pts3d: Numpy array of shape (N,3) giving 3D world coordinates

    Returns:
    - float: a average distance you calculated (threshold is 0.01)

    """
    M = find_projection(pts2d, pts3d)
    pts2d = np.asarray(pts2d, dtype=float)
    pts3d = np.asarray(pts3d, dtype=float)
    X = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
    projected = (M @ X.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    distance = np.mean(np.linalg.norm(projected - pts2d, axis=1))
    return distance

if __name__ == '__main__':
    pts2d = np.loadtxt("task1/pts2d.txt")
    pts3d = np.loadtxt("task1/pts3d.txt")

    # Alternately, for some of the data, we provide pts1/pts1_3D, which you
    # can check your system on via
    """
    data = np.load("task23/ztrans/data.npz")
    pts2d = data['pts1']
    pts3d = data['pts1_3D']
    """
   
    foundDistance = compute_distance(pts2d, pts3d)
    print("Distance: %f" % foundDistance)
