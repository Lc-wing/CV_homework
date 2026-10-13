"""Correctness checks for the Assignment 2 code, run on the CI machine."""
import os
import re
import sys

import numpy as np
import scipy.ndimage
import scipy.signal

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "resources")
sys.path.insert(0, RES)
os.chdir(RES)

import blob_detection  # noqa: E402
import corners  # noqa: E402
import filters  # noqa: E402
from common import read_img  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print("[%s] %s: %s" % ("PASS" if ok else "FAIL", name, detail))


rng = np.random.default_rng(0)
img = read_img("./grace_hopper.png")

# (a) convolve() against scipy.ndimage.convolve with zero padding.
rand_img = rng.standard_normal((37, 53))
for shape in [(3, 3), (1, 3), (3, 1), (5, 5), (9, 9), (1, 1), (2, 2), (4, 3), (2, 5)]:
    k = rng.standard_normal(shape)
    err = np.abs(filters.convolve(rand_img, k)
                 - scipy.ndimage.convolve(rand_img, k, mode="constant", cval=0.0)).max()
    check("convolve vs scipy, kernel %dx%d" % shape, err < 1e-9, "max abs diff = %.3e" % err)
kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                        [0, 2, 3, 5, 5, 5, 3, 2, 0],
                        [3, 3, 5, 3, 0, 3, 5, 3, 3],
                        [2, 5, 3, -12, -23, -12, 3, 5, 2],
                        [2, 5, 0, -23, -40, -23, 0, 5, 2],
                        [2, 5, 3, -12, -23, -12, 3, 5, 2],
                        [3, 3, 5, 3, 0, 3, 5, 3, 3],
                        [0, 2, 3, 5, 5, 5, 3, 2, 0],
                        [0, 0, 3, 2, 2, 2, 3, 0, 0]])
err = np.abs(filters.convolve(img, kernel_LoG2)
             - scipy.ndimage.convolve(img, kernel_LoG2.astype(float), mode="constant")).max()
check("convolve vs scipy, grace_hopper with kernel_LoG2", err < 1e-6, "max abs diff = %.3e" % err)
asym = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
corr = scipy.ndimage.correlate(rand_img, asym, mode="constant")
err_corr = np.abs(filters.convolve(rand_img, asym) - corr).max()
check("convolve is not cross-correlation (asymmetric kernel differs)", err_corr > 1e-3,
      "max |convolve - correlate| = %.3e" % err_corr)
src = open("filters.py", encoding="utf-8").read()
check("filters.py does not import scipy", re.search(r"^\s*(import|from)\s+scipy", src, re.M) is None)

# (b) Gaussian separability.
G = filters.gaussian_kernel(3, 0.572)
g = filters.gaussian_kernel_1d(3, 0.572)
check("2D Gaussian = Gy @ Gx", np.allclose(G, g.T @ g), "max diff = %.3e" % np.abs(G - g.T @ g).max())
sep = filters.convolve(filters.convolve(img, g.T), g)
err = np.abs(filters.convolve(img, G) - sep).max()
check("I*G = (I*Gy)*Gx on the full image", err < 1e-9, "max abs diff = %.3e" % err)
print("    kernel sum sigma=0.572: %.6f, sigma=2: %.6f"
      % (G.sum(), filters.gaussian_kernel(3, 2.0).sum()))

# (c) Derivative signs.
ramp_x = np.tile(np.arange(10, dtype=float), (8, 1))
Ix, Iy, _ = filters.edge_detection(ramp_x)
check("edge_detection on horizontal ramp: Ix = 2, Iy = 0 (interior)",
      np.allclose(Ix[1:-1, 1:-1], 2) and np.allclose(Iy[1:-1, 1:-1], 0),
      "Ix interior unique = %s" % np.unique(Ix[1:-1, 1:-1]))
ramp_y = ramp_x.T.copy()
Ix, Iy, _ = filters.edge_detection(ramp_y)
check("edge_detection on vertical ramp: Iy = 2, Ix = 0 (interior)",
      np.allclose(Iy[1:-1, 1:-1], 2) and np.allclose(Ix[1:-1, 1:-1], 0),
      "Iy interior unique = %s" % np.unique(Iy[1:-1, 1:-1]))

# (d) Sobel versus Gaussian then derivative.
GS = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
kx = np.array([[1, 0, -1]])
full = scipy.signal.convolve2d(GS, kx, mode="full")
expected = np.array([[1, 2, 0, -2, -1], [2, 4, 0, -4, -2], [1, 2, 0, -2, -1]])
check("GS * kx (full)", np.array_equal(full, expected), "\n%s" % full)
Sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
check("Sx = [1,2,1]^T * [1,0,-1]",
      np.array_equal(scipy.signal.convolve2d(np.array([[1], [2], [1]]), kx), Sx))
Gx, Gy, mag = filters.sobel_operator(img)
smooth_dx = filters.convolve(filters.convolve(img, GS), kx)
c = np.corrcoef(Gx[2:-2, 2:-2].ravel(), smooth_dx[2:-2, 2:-2].ravel())[0, 1]
check("corr(I*Sx, (I*GS)*kx) > 0.9", c > 0.9, "corr = %.4f" % c)

# (e) Harris: convolution version against the brute-force loops.
crop = img[100:140, 100:140]
fast = corners.harris_detector(crop)
slow = corners.harris_detector_bruteforce(crop)
check("harris fast vs brute force (grace_hopper crop 40x40)",
      np.allclose(fast, slow, rtol=1e-6, atol=1e-6), "max abs diff = %.3e" % np.abs(fast - slow).max())
r = rng.random((30, 30)) * 255
fast = corners.harris_detector(r)
slow = corners.harris_detector_bruteforce(r)
check("harris fast vs brute force (random 30x30)",
      np.allclose(fast, slow, rtol=1e-6, atol=1e-6), "max abs diff = %.3e" % np.abs(fast - slow).max())

# (f) corner_score against a direct SSD with I = 0 outside the image.
def ssd(image, r0, c0, u, v, win):
    H, W = image.shape
    val = lambda rr, cc: image[rr, cc] if 0 <= rr < H and 0 <= cc < W else 0.0
    h, w = win
    total = 0.0
    for dr in range(-(h // 2), h // 2 + 1):
        for dc in range(-(w // 2), w // 2 + 1):
            rr, cc = r0 + dr, c0 + dc
            total += (val(rr + v, cc + u) - val(rr, cc)) ** 2
    return total


small = img[:60, :70]
for u, v in [(5, 5), (0, 5), (-5, 0)]:
    score = corners.corner_score(small, u, v, (5, 5))
    pts = [(0, 0), (59, 69), (2, 67)] + [tuple(p) for p in rng.integers(0, 60, (17, 2))]
    err = max(abs(score[p] - ssd(small, p[0], p[1], u, v, (5, 5))) for p in pts)
    check("corner_score vs brute-force SSD, (u, v) = (%d, %d)" % (u, v), err < 1e-6,
          "max abs diff = %.3e" % err)

# (g) Task 5 Gaussian against scipy's reference.
cell = blob_detection.read_gray(os.path.join("cells", "002cell.png"))
for sigma in (1.5, 2.5, 4.0):
    ours = blob_detection.gaussian_filter(cell, sigma)
    ref = scipy.ndimage.gaussian_filter(cell.astype(np.float64), sigma, mode="reflect", truncate=3.0)
    err = np.abs(ours - ref).max()
    ksum = blob_detection.gaussian_kernel(sigma).sum()
    check("blob gaussian_filter vs scipy, sigma = %.1f" % sigma, err < 1e-3,
          "max abs diff = %.3e, kernel sum = %.12f" % (err, ksum))

# (h) DoG sign at a bright blob.
yy, xx = np.mgrid[:64, :64]
blob = np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / (2 * 3.0 ** 2))
resp = blob_detection.difference_of_gaussians(blob, 2.5, 4.0)
check("DoG positive at bright blob center", resp[32, 32] > 0,
      "response = %.4f, argmax = %s" % (resp[32, 32], np.unravel_index(resp.argmax(), resp.shape)))
m = blob_detection.find_blob_maxima(resp, 9, 0.3)
check("one maximum on a single synthetic blob", m.sum() == 1 and m[32, 32],
      "maxima at %s" % list(zip(*np.nonzero(m))))

# (i) Task 5 restrictions.
src = open("blob_detection.py", encoding="utf-8").read()
code = re.sub(r'"""[\s\S]*?"""', "", src)
check("no scipy gaussian_filter call in blob_detection.py",
      re.search(r"ndimage\.gaussian_filter\s*\(", code) is None)
check("no library blob detector / cell counter",
      re.search(r"blob_log|blob_dog|blob_doh|peak_local_max|skimage|cv2|SimpleBlobDetector", code) is None)
conv_lines = [line for line in code.splitlines() if "ndimage.convolve(" in line]
check("every ndimage.convolve in blob_detection.py uses mode='reflect'",
      len(conv_lines) > 0 and all('mode="reflect"' in c or "mode='reflect'" in c for c in conv_lines),
      "%d call(s)" % len(conv_lines))

print("\nSUMMARY: %d / %d checks passed" % (sum(results), len(results)))
sys.exit(0 if all(results) else 1)
