import os

import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use zero padding (Set mode = 'constant'). Refer docs for further info.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import read_img, save_img


def corner_score(image, u=5, v=5, window_size=(5, 5)):
    """
    Given an input image, x_offset, y_offset, and window_size,
    return the function E(u,v) for window size W
    corner detector score for that pixel.
    Use zero-padding to handle window values outside of the image.

    Input- image: H x W
           u: a scalar for x offset
           v: a scalar for y offset
           window_size: a tuple for window size

    Output- results: a image of size H x W
    """
    H, W = image.shape
    ph, pw = window_size[0] // 2, window_size[1] // 2
    pu, pv = abs(u), abs(v)
    padded = np.pad(image, ((ph + pv, ph + pv), (pw + pu, pw + pu)), mode='constant')
    # Both grids cover the image extended by half a window on each side;
    # shifted[y, x] = I(x + u, y + v) with I = 0 outside the image.
    base = padded[pv:pv + H + 2 * ph, pu:pu + W + 2 * pw]
    shifted = padded[pv + v:pv + v + H + 2 * ph, pu + u:pu + u + W + 2 * pw]
    diff_sq = (shifted - base) ** 2
    window = np.ones(window_size)
    output = scipy.ndimage.convolve(diff_sq, window, mode='constant', cval=0.0)
    return output[ph:ph + H, pw:pw + W]


def gaussian_window(window_size, sigma=None):
    """Normalized 2D Gaussian weights of shape window_size."""
    h, w = window_size
    if sigma is None:
        sigma = max(h, w) / 5.0
    y = np.arange(h) - (h - 1) / 2.0
    x = np.arange(w) - (w - 1) / 2.0
    xx, yy = np.meshgrid(x, y)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def harris_detector(image, window_size=(5, 5), alpha=0.05, sigma=None):
    """
    Given an input image, calculate the Harris Detector score for all pixels
    You can use same-padding for intensity (or 0-padding for derivatives)
    to handle window values outside of the image.

    Input- image: H x W
    Output- results: a image of size H x W
    """
    image = np.asarray(image, dtype=float)
    # compute the derivatives
    kx = np.array([[1, 0, -1]]) / 2.0
    ky = np.array([[1], [0], [-1]]) / 2.0
    Ix = scipy.ndimage.convolve(image, kx, mode='constant', cval=0.0)
    Iy = scipy.ndimage.convolve(image, ky, mode='constant', cval=0.0)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # For each image location, construct the structure tensor and calculate
    # the Harris response
    # M = sum_w w(x, y) [[Ixx, Ixy], [Ixy, Iyy]]; each entry is a Gaussian-weighted window sum.
    weights = gaussian_window(window_size, sigma)
    M = np.stack([
        scipy.ndimage.convolve(Ixx, weights, mode='constant', cval=0.0),
        scipy.ndimage.convolve(Ixy, weights, mode='constant', cval=0.0),
        scipy.ndimage.convolve(Iyy, weights, mode='constant', cval=0.0),
    ], axis=-1)

    det_M = M[..., 0] * M[..., 2] - M[..., 1] ** 2
    trace_M = M[..., 0] + M[..., 2]
    response = det_M - alpha * trace_M ** 2

    return response


def harris_detector_bruteforce(image, window_size=(5, 5), alpha=0.05, sigma=None):
    """Slow reference version that sums the structure tensor with explicit loops."""
    image = np.asarray(image, dtype=float)
    kx = np.array([[1, 0, -1]]) / 2.0
    ky = np.array([[1], [0], [-1]]) / 2.0
    Ix = scipy.ndimage.convolve(image, kx, mode='constant', cval=0.0)
    Iy = scipy.ndimage.convolve(image, ky, mode='constant', cval=0.0)
    weights = gaussian_window(window_size, sigma)
    h, w = window_size
    H, W = image.shape
    response = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            M = np.zeros((2, 2))
            for di in range(h):
                for dj in range(w):
                    y = i + di - (h - 1) // 2
                    x = j + dj - (w - 1) // 2
                    if 0 <= y < H and 0 <= x < W:
                        g = np.array([Ix[y, x], Iy[y, x]])
                        M += weights[di, dj] * np.outer(g, g)
            response[i, j] = np.linalg.det(M) - alpha * np.trace(M) ** 2
    return response


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # -- Corner Score (warm-up, not graded in Assignment2.pdf) --
    # (a): Complete corner_score()

    # (b)
    # Define offsets and window size and calulcate corner score
    u, v, W = 5, 5, (5, 5)

    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score.png")

    # Computing the corner scores for various u, v values.
    score = corner_score(img, 0, 5, W)
    save_img(score, "./feature_detection/corner_score05.png")

    score = corner_score(img, 0, -5, W)
    save_img(score, "./feature_detection/corner_score0-5.png")

    score = corner_score(img, 5, 0, W)
    save_img(score, "./feature_detection/corner_score50.png")

    score = corner_score(img, -5, 0, W)
    save_img(score, "./feature_detection/corner_score-50.png")

    # -- TODO Task 4: Harris Corner Detector --
    # (a): Complete harris_detector()

    # (b)
    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")
    print("Task 4(b): Harris response min = %.4f, max = %.4f"
          % (harris_corners.min(), harris_corners.max()))

    plt.figure(figsize=(7, 6))
    plt.imshow(harris_corners, cmap='jet')
    plt.colorbar(label='Harris response R')
    plt.title('Harris corner response (alpha = 0.05, 5x5 Gaussian window)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("./feature_detection/harris_heatmap.png", dpi=150)
    plt.close()
    print("./feature_detection/harris_heatmap.png is saved!")

    # Strong corners: local maxima of R above 1% of the maximum response.
    local_max = scipy.ndimage.maximum_filter(harris_corners, size=7) == harris_corners
    corners = local_max & (harris_corners > 0.01 * harris_corners.max())
    rows, cols = np.nonzero(corners)
    print("Task 4(b): %d corners above 1%% of max response" % len(rows))
    plt.figure(figsize=(7, 6))
    plt.imshow(img, cmap='gray')
    plt.scatter(cols, rows, s=6, c='r')
    plt.title('Harris corners (local maxima, R > 0.01 max R)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("./feature_detection/harris_corners_overlay.png", dpi=150)
    plt.close()
    print("./feature_detection/harris_corners_overlay.png is saved!")


if __name__ == "__main__":
    main()
