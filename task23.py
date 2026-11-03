from utils import homogenize, draw_epipolar, visualize_pcd
import numpy as np
import cv2
import pdb
import os


def find_fundamental_matrix(shape, pts1, pts2):
    """
    Computes Fundamental Matrix F that relates points in two images by the:

        [u' v' 1] F [u v 1]^T = 0
        or
        l = F [u v 1]^T  -- the epipolar line for point [u v] in image 2
        [u' v' 1] F = l'   -- the epipolar line for point [u' v'] in image 1

    Where (u,v) and (u',v') are the 2D image coordinates of the left and
    the right images respectively.

    Inputs:
    - shape: Tuple containing shape of img1
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    """

    #This will give you an answer you can compare with
    #Your answer should match closely once you've divided by the last entry
    _FOpenCV, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

    pts1 = np.asarray(pts1, dtype=float)
    pts2 = np.asarray(pts2, dtype=float)

    def normalizing_transform(pts):
        """Center the points at 0 and scale them to mean distance sqrt(2).

        shape gives the image size. The translation uses the point centroid,
        and the scale uses the image diagonal so coordinates from a large
        image are brought near unit size (Hartley normalization, with the
        image diagonal as a reference when the point spread is degenerate).
        """
        center = pts.mean(axis=0)
        mean_dist = np.sqrt(((pts - center) ** 2).sum(axis=1)).mean()
        height, width = shape[:2]
        image_scale = 1.0 / np.sqrt(height ** 2 + width ** 2)
        scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else image_scale
        return np.array([[scale, 0.0, -scale * center[0]],
                         [0.0, scale, -scale * center[1]],
                         [0.0, 0.0, 1.0]])

    T1 = normalizing_transform(pts1)
    T2 = normalizing_transform(pts2)
    p1 = (T1 @ homogenize(pts1).T).T
    p2 = (T2 @ homogenize(pts2).T).T
    # One row per correspondence: [u', v', 1] kron [u, v, 1].
    design = np.einsum("ni,nj->nij", p2, p1).reshape(pts1.shape[0], 9)
    _, _, vt = np.linalg.svd(design)
    F = vt[-1].reshape(3, 3)
    # Closest rank-2 matrix.
    u_mat, singular, vt = np.linalg.svd(F)
    singular[-1] = 0.0
    F = u_mat @ np.diag(singular) @ vt
    F = T2.T @ F @ T1
    return F


def compute_epipoles(F):
    """
    Given a Fundamental Matrix F, return the epipoles represented in
    homogeneous coordinates.

    Check: e2@F and F@e1 should be close to [0,0,0]

    Inputs:
    - F: the fundamental matrix

    Return:
    - e1: the epipole for image 1 in homogeneous coordinates
    - e2: the epipole for image 2 in homogeneous coordinates
    """
    # e1 is the right null vector (F e1 = 0); e2 is the left null vector.
    u_mat, _, vt = np.linalg.svd(F)
    e1 = vt[-1]
    e2 = u_mat[:, -1]
    return e1, e2


def find_triangulation(K1, K2, F, pts1, pts2):
    """
    Extracts 3D points from 2D points and camera matrices. Let X be a
    point in 3D in homogeneous coordinates. For two cameras, we have

        p1 === M1 X
        p2 === M2 X

    Triangulation is to solve for X given p1, p2, M1, M2.

    Inputs:
    - K1: Numpy array of shape (3,3) giving camera instrinsic matrix for img1
    - K2: Numpy array of shape (3,3) giving camera instrinsic matrix for img2
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - pcd: Numpy array of shape (N,4) giving the homogeneous 3D point cloud
      data
    """
    K1 = np.asarray(K1, dtype=float)
    K2 = np.asarray(K2, dtype=float)
    pts1 = np.asarray(pts1, dtype=float)
    pts2 = np.asarray(pts2, dtype=float)
    essential = K2.T @ F @ K1
    u_mat, _, vt = np.linalg.svd(essential)
    if np.linalg.det(u_mat) < 0:
        u_mat = -u_mat
    if np.linalg.det(vt) < 0:
        vt = -vt
    W = np.array([[0.0, -1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    candidates = [
        (u_mat @ W @ vt, u_mat[:, 2]),
        (u_mat @ W @ vt, -u_mat[:, 2]),
        (u_mat @ W.T @ vt, u_mat[:, 2]),
        (u_mat @ W.T @ vt, -u_mat[:, 2]),
    ]

    def linear_triangulation(M1, M2):
        points = []
        for (u, v), (uu, vv) in zip(pts1, pts2):
            design = np.vstack([
                u * M1[2] - M1[0],
                v * M1[2] - M1[1],
                uu * M2[2] - M2[0],
                vv * M2[2] - M2[1],
            ])
            _, _, point_vt = np.linalg.svd(design)
            point = point_vt[-1]
            points.append(point / point[3])
        return np.asarray(points)

    best = None
    for rotation, translation in candidates:
        if np.linalg.det(rotation) < 0:
            rotation = -rotation
            translation = -translation
        M1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        M2 = K2 @ np.hstack([rotation, translation.reshape(3, 1)])
        cloud = linear_triangulation(M1, M2)
        depth1 = cloud[:, 2]
        depth2 = (rotation @ cloud[:, :3].T).T + translation
        in_front = np.sum((depth1 > 0) & (depth2[:, 2] > 0))
        if best is None or in_front > best[0]:
            best = (in_front, cloud)
    pcd = best[1]
    return pcd


if __name__ == '__main__':

    # You can run it on one or all the examples
    names = os.listdir("task23")
    output = "results/"

    if not os.path.exists(output):
        os.mkdir(output)

    for name in names:
        print(name)

        # load the information
        img1 = cv2.imread(os.path.join("task23", name, "im1.png"))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(os.path.join("task23", name, "im2.png"))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        data = np.load(os.path.join("task23", name, "data.npz"))
        pts1 = data['pts1'].astype(float)
        pts2 = data['pts2'].astype(float)
        K1 = data['K1']
        K2 = data['K2']
        shape = img1.shape

        # compute F
        F = find_fundamental_matrix(shape, pts1, pts2)
        # compute the epipoles
        e1, e2 = compute_epipoles(F)
        print(e1, e2)
        #to get the real coordinates, divide by the last entry
        print(e1[:2]/e1[-1], e2[:2]/e2[-1])

        outname = os.path.join(output, name + "_us.png")
        # If filename isn't provided or is None, this plt.shows().
        # If it's provided, it saves it
        draw_epipolar(img1, img2, F, pts1[::10, :], pts2[::10, :],
                      epi1=e1, epi2=e2, filename=outname)

        if 0:
            #you can turn this on or off
            pcd = find_triangulation(K1, K2, F, pts1, pts2)
            visualize_pcd(pcd,
                          filename=os.path.join(output, name + "_rec.png"))


