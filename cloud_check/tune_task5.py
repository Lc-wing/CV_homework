"""Parameter sweeps for Task 5; writes count tables and comparison panels."""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "resources"
sys.path.insert(0, str(RES))
os.chdir(RES)

import blob_detection as bd  # noqa: E402

OUT = HERE / "tuning"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
LABEL_COLORS = rng.random((4096, 3)) * 0.8 + 0.2


def otsu(x, bins=256):
    hist, edges = np.histogram(x.ravel(), bins=bins, range=(0, 1))
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centers)
    var = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0) + 1e-12)
    return centers[np.argmax(var)]


def label_overlay(image, labels, kept):
    rgb = np.dstack([image] * 3) * 0.6
    keep = np.isin(labels, kept)
    rgb[keep] = LABEL_COLORS[labels[keep] % len(LABEL_COLORS)]
    return np.clip(rgb, 0, 1)


def morphology_sweep(name):
    image = bd.read_gray(bd.CELL_DIR / name)
    t_otsu = otsu(image)
    print(f"\n== {name}: min {image.min():.3f} max {image.max():.3f} "
          f"p50 {np.percentile(image, 50):.3f} p90 {np.percentile(image, 90):.3f} otsu {t_otsu:.3f}")
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    print("threshold | erosion=dilation 0..3 (closing 1, area 12-400) | median area at erosion 1")
    for t in thresholds:
        row = []
        med = 0
        for e in range(4):
            mask = bd.morphology_mask(image, t, e, e, 1)
            n, labels, kept = bd.count_components(mask, 12, 400)
            row.append(n)
            if e == 1:
                areas = np.bincount(labels.ravel())[1:]
                med = np.median(areas) if len(areas) else 0
                big = int((areas > 400).sum())
        print(f"  {t:.2f}     | {row} | median area {med:.0f}, >400: {big}")

    print("global threshold 0.50-0.70 | erosion=dilation 0..3")
    for t in [0.50, 0.55, 0.60, 0.65, 0.70]:
        row = [bd.count_components(bd.morphology_mask(image, t, e, e, 1), 12, 400)[0] for e in range(4)]
        print(f"  {t:.2f}     | {row}")

    for bg in (10, 15, 25):
        flat = image - bd.gaussian_filter(image, bg)
        print(f"background_sigma {bg}: flattened p50 {np.percentile(flat, 50):.3f} "
              f"p75 {np.percentile(flat, 75):.3f} p90 {np.percentile(flat, 90):.3f} "
              f"max {flat.max():.3f} otsu-ish {otsu(np.clip(flat + 0.5, 0, 1)) - 0.5:.3f}")
        print("  threshold | erosion=dilation 0..3 (closing 1, area 12-400) | median area, >400 at e=1")
        for t in [0.0, 0.03, 0.06, 0.09, 0.12, 0.15]:
            row = []
            for e in range(4):
                mask = bd.morphology_mask(image, t, e, e, 1, bg)
                n, labels, kept = bd.count_components(mask, 12, 400)
                row.append(n)
                if e == 1:
                    areas = np.bincount(labels.ravel())[1:]
                    med = np.median(areas) if len(areas) else 0
                    big = int((areas > 400).sum())
                    small = int((areas < 12).sum())
            print(f"    {t:.2f}    | {row} | median {med:.0f}, >400: {big}, <12: {small}")

    bg = 15
    show_t = [0.03, 0.06, 0.09, 0.12]
    show_e = [0, 1, 2, 3]
    fig, axs = plt.subplots(len(show_t), len(show_e), figsize=(4 * len(show_e), 4 * len(show_t)))
    for i, t in enumerate(show_t):
        for j, e in enumerate(show_e):
            mask = bd.morphology_mask(image, t, e, e, 1, bg)
            n, labels, kept = bd.count_components(mask, 12, 400)
            axs[i, j].imshow(label_overlay(image, labels, kept), interpolation="nearest")
            axs[i, j].set_title(f"bg={bg} t={t:.2f} e=d={e}: {n}")
            axs[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(OUT / f"morph_{Path(name).stem}.png", dpi=80)
    plt.close()


def dog_sweep(name):
    path = bd.CELL_DIR / name
    image = bd.read_gray(path)
    pairs = [(1.5, 2.4), (2.0, 3.2), (2.5, 4.0), (3.0, 4.8), (3.5, 5.6), (2.0, 4.0)]
    print(f"\n== DoG sweep on {name} (neighborhood 9, threshold_rel 0.3)")
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (s1, s2) in zip(axs.ravel(), pairs):
        resp = bd.difference_of_gaussians(image, s1, s2)
        m = bd.find_blob_maxima(resp, 9, 0.3)
        print(f"  sigma ({s1}, {s2}): {int(m.sum())} maxima, response max {resp.max():.4f}")
        ax.imshow(image, cmap="gray")
        for r, c in zip(*np.nonzero(m)):
            ax.add_patch(Circle((c, r), np.sqrt(2) * s1, fill=False, color="yellow", lw=0.8))
        ax.set_title(f"sigma=({s1},{s2}): {int(m.sum())}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / f"dog_pairs_{Path(name).stem}.png", dpi=80)
    plt.close()

    print("  threshold_rel x neighborhood (sigma 2.0, 3.2 / 2.5, 4.0):")
    for s1, s2 in [(2.0, 3.2), (2.5, 4.0)]:
        resp = bd.difference_of_gaussians(image, s1, s2)
        for nb in (5, 7, 9, 11):
            row = [int(bd.find_blob_maxima(resp, nb, rel).sum()) for rel in (0.05, 0.1, 0.2, 0.3, 0.4)]
            print(f"    sigma ({s1},{s2}) nb {nb:2d}: rel 0.05/0.1/0.2/0.3/0.4 -> {row}")


def zoom(name):
    image = np.asarray(plt.imread(bd.CELL_DIR / name))[..., :3]
    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    for ax, (r, c) in zip(axs.ravel(), [(0, 0), (0, 128), (128, 0), (128, 128)]):
        crop = image[r:r + 128, c:c + 128]
        ax.imshow(np.clip(crop * 2.5, 0, 1), interpolation="nearest",
                  extent=(c, c + 128, r + 128, r))
        ax.set_xticks(range(c, c + 129, 32))
        ax.set_yticks(range(r, r + 129, 32))
        ax.grid(color="white", alpha=0.4)
        ax.set_title(f"rows {r}-{r + 127}, cols {c}-{c + 127}")
    plt.tight_layout()
    plt.savefig(OUT / f"zoom_{Path(name).stem}.png", dpi=70)
    plt.close()


def dog_zoom(name):
    image = bd.read_gray(bd.CELL_DIR / name)
    pairs = [(2.0, 3.2), (2.5, 4.0), (3.0, 4.8)]
    quads = [(0, 0), (128, 128)]
    fig, axs = plt.subplots(len(quads), len(pairs), figsize=(6 * len(pairs), 6 * len(quads)))
    for j, (s1, s2) in enumerate(pairs):
        m = bd.find_blob_maxima(bd.difference_of_gaussians(image, s1, s2), 9, 0.3)
        for i, (r0, c0) in enumerate(quads):
            ax = axs[i, j]
            ax.imshow(image[r0:r0 + 128, c0:c0 + 128], cmap="gray", interpolation="nearest",
                      extent=(c0, c0 + 128, r0 + 128, r0))
            rows, cols = np.nonzero(m[r0:r0 + 128, c0:c0 + 128])
            for r, c in zip(rows, cols):
                ax.add_patch(Circle((c0 + c + 0.5, r0 + r + 0.5), np.sqrt(2) * s1,
                                    fill=False, color="yellow", lw=1))
                ax.plot(c0 + c + 0.5, r0 + r + 0.5, "r.", ms=3)
            ax.set_title(f"sigma=({s1},{s2}) rows {r0}+ cols {c0}+: {len(rows)}")
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / f"dog_zoom_{Path(name).stem}.png", dpi=60)
    plt.close()


if __name__ == "__main__":
    for name in bd.SELECTED:
        morphology_sweep(name)
    dog_sweep(bd.SELECTED[0])
    zoom(bd.SELECTED[0])
    dog_zoom(bd.SELECTED[0])
