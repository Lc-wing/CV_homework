"""Report-ready composite figures and a JSON file with every number used in the report."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import scipy.ndimage

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "resources"
sys.path.insert(0, str(RES))
os.chdir(RES)

import blob_detection as bd  # noqa: E402
import corners  # noqa: E402
import filters  # noqa: E402
from common import read_img  # noqa: E402

FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
results = {}


def stats(a):
    return {"min": float(a.min()), "max": float(a.max()),
            "range": float(a.max() - a.min()), "mean": float(a.mean())}


def panel(images, titles, path, cmap="gray", vmins=None, vmaxs=None, colorbar=True, size=4.6):
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(size * n, size * 0.95))
    axs = np.atleast_1d(axs)
    for i, (ax, im, title) in enumerate(zip(axs, images, titles)):
        h = ax.imshow(im, cmap=cmap, vmin=None if vmins is None else vmins[i],
                      vmax=None if vmaxs is None else vmaxs[i])
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        if colorbar:
            plt.colorbar(h, ax=ax, fraction=0.046, pad=0.03)
    plt.tight_layout()
    plt.savefig(FIG / path, dpi=150, bbox_inches="tight")
    plt.close()


img = read_img("./grace_hopper.png")

# Task 1
G = filters.gaussian_kernel(3, 0.572)
g1 = filters.gaussian_kernel_1d(3, 0.572)
G_norm = filters.gaussian_kernel(3, 0.572, normalize=True)
G2 = filters.gaussian_kernel(3, 2.0)
blur = filters.convolve(img, G)
blur_norm = filters.convolve(img, G_norm)
blur2 = filters.convolve(img, G2)
results["t1"] = {
    "kernel": np.round(G, 4).tolist(),
    "kernel_1d": np.round(g1.ravel(), 4).tolist(),
    "sep_kernel_err": float(np.abs(G - g1.T @ g1).max()),
    "sep_image_err": float(np.abs(blur - filters.convolve(filters.convolve(img, g1.T), g1)).max()),
    "sum_0572": float(G.sum()),
    "sum_2": float(G2.sum()),
    "kernel_2": np.round(G2, 4).tolist(),
    "orig": stats(img), "raw_0572": stats(blur), "norm_0572": stats(blur_norm), "raw_2": stats(blur2),
}
panel([img, blur], ["Original grace_hopper.png", "Gaussian 3x3, sigma = 0.572"],
      "t1c_gaussian.png", vmins=[0, 0], vmaxs=[255, 255])
crop = (slice(130, 260), slice(180, 330))
panel([img[crop], blur[crop]], ["Original (crop)", "Gaussian filtered (crop)"],
      "t1c_gaussian_crop.png", vmins=[0, 0], vmaxs=[255, 255], colorbar=False)
panel([blur_norm, blur2],
      ["sigma = 0.572, kernel sum = 1", "sigma = 2, kernel sum = %.3f" % G2.sum()],
      "t1d_sum.png", vmins=[0, 0], vmaxs=[255, 255])
_, _, e0 = filters.edge_detection(img)
_, _, e1 = filters.edge_detection(blur)
results["t1"]["edge_orig"] = stats(e0)
results["t1"]["edge_blur"] = stats(e1)
vmax = float(np.percentile(e0, 99.5))
panel([e0, e1], ["Gradient magnitude: original", "Gradient magnitude: Gaussian filtered"],
      "t1g_edges.png", vmins=[0, 0], vmaxs=[vmax, vmax])
panel([e0[crop], e1[crop]], ["Original (crop)", "Gaussian filtered (crop)"],
      "t1g_edges_crop.png", vmins=[0, 0], vmaxs=[vmax, vmax], colorbar=False)

# Task 2
Gx, Gy, mag = filters.sobel_operator(img)
GS = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
kx = np.array([[1, 0, -1]])
smooth_dx = filters.convolve(filters.convolve(img, GS), kx)
results["t2"] = {
    "corr": float(np.corrcoef(Gx[2:-2, 2:-2].ravel(), smooth_dx[2:-2, 2:-2].ravel())[0, 1]),
    "ratio": float(np.dot(smooth_dx[2:-2, 2:-2].ravel(), Gx[2:-2, 2:-2].ravel())
                   / np.dot(Gx[2:-2, 2:-2].ravel(), Gx[2:-2, 2:-2].ravel())),
    "Gx": stats(Gx), "Gy": stats(Gy), "mag": stats(mag),
}
lim = float(np.percentile(np.abs(Gx), 99.5))
panel([Gx, Gy, mag], ["I * Sx", "I * Sy", "Gradient magnitude"], "t2c_sobel.png",
      vmins=[-lim, -lim, 0], vmaxs=[lim, lim, float(np.percentile(mag, 99.5))])
lim2 = float(np.percentile(np.abs(smooth_dx), 99.5))
panel([Gx, smooth_dx], ["I * Sx", "(I * GS) * kx"], "t2a_compare.png",
      vmins=[-lim, -lim2], vmaxs=[lim, lim2])

# Task 3
k1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
k2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
               [0, 2, 3, 5, 5, 5, 3, 2, 0],
               [3, 3, 5, 3, 0, 3, 5, 3, 3],
               [2, 5, 3, -12, -23, -12, 3, 5, 2],
               [2, 5, 0, -23, -40, -23, 0, 5, 2],
               [2, 5, 3, -12, -23, -12, 3, 5, 2],
               [3, 3, 5, 3, 0, 3, 5, 3, 3],
               [0, 2, 3, 5, 5, 5, 3, 2, 0],
               [0, 0, 3, 2, 2, 2, 3, 0, 0]])
L1 = filters.convolve(img, k1)
L2 = filters.convolve(img, k2)
results["t3"] = {"L1": stats(L1), "L2": stats(L2), "sum1": int(k1.sum()), "sum2": int(k2.sum())}
l1 = float(np.percentile(np.abs(L1), 99))
l2 = float(np.percentile(np.abs(L2), 99))
panel([L1, L2], ["I * kernel_LoG1 (3x3)", "I * kernel_LoG2 (9x9)"], "t3a_log.png",
      vmins=[-l1, -l2], vmaxs=[l1, l2])
panel([np.abs(L1), np.abs(L2)], ["|I * kernel_LoG1|", "|I * kernel_LoG2|"], "t3a_log_abs.png",
      cmap="magma", vmins=[0, 0], vmaxs=[l1, l2])
panel([L1[crop], L2[crop]], ["LoG1 (crop)", "LoG2 (crop)"], "t3a_log_crop.png",
      vmins=[-l1, -l2], vmaxs=[l1, l2], colorbar=False)
data = np.load("log1d.npz")
dog = data["gauss53"] - data["gauss50"]
scale = float(np.dot(dog, data["log50"]) / np.dot(dog, dog))
results["t3"]["dog_scale"] = scale
results["t3"]["dog_sum"] = float(dog.sum())
results["t3"]["log_sum"] = float(data["log50"].sum())
results["t3"]["dog_rel_err"] = float(np.linalg.norm(scale * dog - data["log50"]) / np.linalg.norm(data["log50"]))
results["t3"]["log50_center"] = float(data["log50"][len(dog) // 2])
results["t3"]["dog_center"] = float(dog[len(dog) // 2])
x = np.arange(len(dog)) - len(dog) // 2
fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
axs[0].plot(x, data["gauss50"], label="Gaussian, sigma = 50")
axs[0].plot(x, data["gauss53"], label="Gaussian, sigma = 53")
axs[0].set_title("(a) Two Gaussians")
axs[1].plot(x, dog, color="C2", label="gauss53 - gauss50")
axs[1].set_title("(b) Difference of Gaussians")
axs[2].plot(x, data["log50"], color="C3", lw=2.5, label="LoG, sigma = 50")
axs[2].plot(x, scale * dog, "k--", label="DoG x %.3f" % scale)
axs[2].set_title("(c) LoG vs scaled DoG")
for ax in axs:
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlabel("x")
plt.tight_layout()
plt.savefig(FIG / "t3b_dog_log.png", dpi=150)
plt.close()

# Task 4
R = corners.harris_detector(img)
local_max = scipy.ndimage.maximum_filter(R, size=7) == R
pts = local_max & (R > 0.01 * R.max())
results["t4"] = {"R": stats(R), "n_corners": int(pts.sum()),
                 "frac_negative": float((R < 0).mean()),
                 "bf_err_crop": float(np.abs(corners.harris_detector(img[100:140, 100:140])
                                             - corners.harris_detector_bruteforce(img[100:140, 100:140])).max())}
fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
lim = float(np.percentile(np.abs(R), 99.7))
h = axs[0].imshow(R, cmap="jet", vmin=-lim, vmax=lim)
axs[0].set_title("Harris response R (clipped at 99.7th percentile of |R|)")
axs[0].axis("off")
plt.colorbar(h, ax=axs[0], fraction=0.046, pad=0.03)
axs[1].imshow(img, cmap="gray")
rr, cc = np.nonzero(pts)
axs[1].scatter(cc, rr, s=7, c="r")
axs[1].set_title("Local maxima of R with R > 0.01 max R (%d)" % len(rr))
axs[1].axis("off")
plt.tight_layout()
plt.savefig(FIG / "t4_harris.png", dpi=150, bbox_inches="tight")
plt.close()
h = plt.figure(figsize=(6.5, 5.6))
plt.imshow(R, cmap="jet")
plt.colorbar(fraction=0.046, pad=0.03)
plt.title("Harris response R (full range)")
plt.axis("off")
plt.tight_layout()
plt.savefig(FIG / "t4_harris_full.png", dpi=150, bbox_inches="tight")
plt.close()

# Task 5(a)
name = bd.SELECTED[0]
path = bd.CELL_DIR / name
gray = bd.read_gray(path)
rgb = np.asarray(Image.open(path).convert("RGB"))
s1, s2 = bd.DOG_SIGMAS
resp = bd.difference_of_gaussians(gray, s1, s2)
maxima = bd.find_blob_maxima(resp, bd.DOG_NEIGHBORHOOD, bd.DOG_THRESHOLD_REL)
results["t5a"] = {"image": name, "sigmas": [s1, s2], "neighborhood": bd.DOG_NEIGHBORHOOD,
                  "threshold_rel": bd.DOG_THRESHOLD_REL, "resp_max": float(resp.max()),
                  "abs_threshold": float(bd.DOG_THRESHOLD_REL * resp.max()),
                  "count": int(maxima.sum()), "pairs": []}


def draw_blobs(ax, maxima, radius, title):
    ax.imshow(rgb)
    for r, c in zip(*np.nonzero(maxima)):
        ax.add_patch(Circle((c, r), radius, fill=False, color="yellow", lw=0.9))
    ax.set_title(title, fontsize=11)
    ax.axis("off")


fig, axs = plt.subplots(1, 3, figsize=(16, 5.6))
axs[0].imshow(rgb)
axs[0].set_title("Input %s" % name)
axs[0].axis("off")
h = axs[1].imshow(resp, cmap="magma")
axs[1].set_title("DoG response G(%.1f) - G(%.1f)" % (s1, s2))
axs[1].axis("off")
plt.colorbar(h, ax=axs[1], fraction=0.046, pad=0.03)
draw_blobs(axs[2], maxima, np.sqrt(2) * s1, "Detected maxima: %d" % maxima.sum())
plt.tight_layout()
plt.savefig(FIG / "task5a_panel.png", dpi=150, bbox_inches="tight")
plt.close()

fig, axs = plt.subplots(1, len(bd.SCALE_PAIRS), figsize=(4.4 * len(bd.SCALE_PAIRS), 4.6))
for ax, (a, b) in zip(axs, bd.SCALE_PAIRS):
    m = bd.find_blob_maxima(bd.difference_of_gaussians(gray, a, b), bd.DOG_NEIGHBORHOOD, bd.DOG_THRESHOLD_REL)
    results["t5a"]["pairs"].append({"sigmas": [a, b], "count": int(m.sum())})
    draw_blobs(ax, m, np.sqrt(2) * a, "sigma = (%.1f, %.1f): %d" % (a, b, m.sum()))
plt.tight_layout()
plt.savefig(FIG / "task5a_scales.png", dpi=150, bbox_inches="tight")
plt.close()

zoom = (slice(0, 128), slice(0, 128))
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(np.clip(rgb[zoom].astype(float) * 2.2 / 255, 0, 1), interpolation="nearest")
axs[0].set_title("Input, top-left 128 x 128 (brightened)")
axs[0].axis("off")
axs[1].imshow(np.clip(rgb[zoom].astype(float) * 2.2 / 255, 0, 1), interpolation="nearest")
for r, c in zip(*np.nonzero(maxima[zoom])):
    axs[1].add_patch(Circle((c, r), np.sqrt(2) * s1, fill=False, color="yellow", lw=1.2))
axs[1].set_title("DoG maxima, sigma = (%.1f, %.1f)" % (s1, s2))
axs[1].axis("off")
plt.tight_layout()
plt.savefig(FIG / "task5a_zoom.png", dpi=150, bbox_inches="tight")
plt.close()

results["t5a"]["rel_sweep"] = {str(r): int(bd.find_blob_maxima(resp, bd.DOG_NEIGHBORHOOD, r).sum())
                               for r in (0.1, 0.2, 0.3, 0.4, 0.5)}
results["t5a"]["nb_sweep"] = {str(nb): int(bd.find_blob_maxima(resp, nb, bd.DOG_THRESHOLD_REL).sum())
                              for nb in (5, 7, 9, 11, 13)}

# Task 5(b)
results["t5b"] = {}
for fname in bd.SELECTED:
    p = bd.IMAGE_PARAMETERS[fname]
    im = bd.read_gray(bd.CELL_DIR / fname)
    mask = bd.morphology_mask(im, p["threshold"], p["erosion_iterations"], p["dilation_iterations"],
                              p["closing_iterations"], p["background_sigma"])
    raw = bd.morphology_mask(im, p["threshold"], 0, 0, 0, p["background_sigma"])
    n, labels, kept = bd.count_components(mask, p["min_area"], p["max_area"])
    areas = np.bincount(labels.ravel())[1:]
    results["t5b"][fname] = {"params": p, "count": int(n), "n_components": int(labels.max()),
                             "rejected_small": int((areas < p["min_area"]).sum()),
                             "rejected_large": int((areas > p["max_area"]).sum()),
                             "median_area_kept": float(np.median(areas[kept - 1])) if n else 0.0}
    src = np.asarray(Image.open(bd.CELL_DIR / fname).convert("RGB"))
    fig, axs = plt.subplots(1, 4, figsize=(19, 5.2))
    axs[0].imshow(src)
    axs[0].set_title("Input %s" % fname)
    axs[1].imshow(raw, cmap="gray")
    axs[1].set_title("Threshold %.2f after flattening" % p["threshold"])
    axs[2].imshow(mask, cmap="gray")
    axs[2].set_title("Cleaned mask (close %d, erode %d, dilate %d)"
                     % (p["closing_iterations"], p["erosion_iterations"], p["dilation_iterations"]))
    axs[3].imshow(src)
    objects = scipy.ndimage.find_objects(labels)
    for lab in kept:
        sl = objects[lab - 1]
        r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        axs[3].add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0,
                                       fill=False, color="red", lw=0.7))
    axs[3].set_title("Retained components: %d (area %d-%d)" % (n, p["min_area"], p["max_area"]))
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG / ("task5b_%s_panel.png" % Path(fname).stem), dpi=130, bbox_inches="tight")
    plt.close()

# Baseline: starter threshold without flattening on the Task 5(a) image.
b = bd.BASELINE_PARAMETERS
bmask = bd.morphology_mask(gray, b["threshold"], b["erosion_iterations"], b["dilation_iterations"],
                           b["closing_iterations"], 0)
bn, blabels, bkept = bd.count_components(bmask, b["min_area"], b["max_area"])
bareas = np.bincount(blabels.ravel())[1:]
results["t5b"]["baseline_" + name] = {"params": b, "count": int(bn),
                                      "largest_area": int(bareas.max()) if len(bareas) else 0}
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(gray > b["threshold"], cmap="gray")
axs[0].set_title("Global threshold %.2f, no flattening" % b["threshold"])
axs[1].imshow(bmask, cmap="gray")
axs[1].set_title("Cleaned: %d components kept (largest %d px)" % (bn, bareas.max()))
for ax in axs:
    ax.axis("off")
plt.tight_layout()
plt.savefig(FIG / "task5b_baseline.png", dpi=130, bbox_inches="tight")
plt.close()

# Task 5(c): one-parameter sweeps on the Task 5(a) image.
p = bd.IMAGE_PARAMETERS[name]


def mcount(**ch):
    q = dict(p, **ch)
    m = bd.morphology_mask(gray, q["threshold"], q["erosion_iterations"], q["dilation_iterations"],
                           q["closing_iterations"], q["background_sigma"])
    return int(bd.count_components(m, q["min_area"], q["max_area"])[0])


ts = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
results["t5c"] = {"threshold_sweep": {str(t): mcount(threshold=t) for t in ts},
                  "erosion_sweep": {str(e): mcount(erosion_iterations=e) for e in (1, 2, 3, 4)},
                  "dog": results["t5a"]["count"], "morph": results["t5b"][name]["count"]}
rels = [0.1, 0.2, 0.3, 0.4, 0.5]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
axs[0].plot(ts, [results["t5c"]["threshold_sweep"][str(t)] for t in ts], "o-")
axs[0].set_xlabel("morphology threshold (after flattening)")
axs[0].set_ylabel("component count")
axs[0].set_title("Morphology: threshold sweep")
axs[1].plot(rels, [results["t5a"]["rel_sweep"][str(r)] for r in rels], "s-", color="C1")
axs[1].set_xlabel("DoG threshold_rel")
axs[1].set_ylabel("maxima count")
axs[1].set_title("DoG: relative threshold sweep")
for ax in axs:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "task5c_sweeps.png", dpi=150, bbox_inches="tight")
plt.close()

with open(HERE / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
