"""Starter code for Task 5: blob detection and morphological cell counting.

The supplied images are read from ``cells/``. Complete the functions marked
TODO, then run this file to process four default images. You may change the
parameters or filenames in ``main`` when preparing your report.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


CELL_DIR = Path("cells")
OUTPUT_DIR = Path("cell_detections")

# 3x3 cross (4-connected) structuring element. Iterating it n times acts like
# a diamond of radius n, which is close to the round shape of a cell.
STRUCTURE = ndimage.generate_binary_structure(2, 1)


def read_gray(path):
    """Read an image as a float32 grayscale array in the range [0, 1]."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.maximum(luminance, rgb[..., 2])


def gaussian_kernel(sigma, truncate=3.0):
    """Return a normalized (2r+1) x (2r+1) Gaussian kernel with r = ceil(truncate * sigma)."""
    radius = max(1, int(np.ceil(truncate * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, x)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_filter(image, sigma):
    """Return a Gaussian-filtered image using reflect padding.

    TODO: construct a normalized 2-D Gaussian kernel and convolve it with
    ``scipy.ndimage.convolve(..., mode="reflect")``. Do not call
    ``scipy.ndimage.gaussian_filter`` for this function.
    """
    kernel = gaussian_kernel(sigma)
    return ndimage.convolve(np.asarray(image, dtype=np.float64), kernel, mode="reflect")


def difference_of_gaussians(image, sigma_small, sigma_large):
    """Return the bright-blob Difference-of-Gaussians response.

    TODO: blur the image at both scales and subtract them in the order that
    gives a positive response at bright cell centers.
    """
    # At the center of a bright blob the narrow blur keeps more of the peak
    # than the wide blur, so G_small - G_large is positive there.
    return gaussian_filter(image, sigma_small) - gaussian_filter(image, sigma_large)


def find_blob_maxima(response, neighborhood_size=9, threshold_rel=0.30):
    """Return an HxW boolean mask containing local DoG maxima.

    TODO: use a maximum filter, a relative response threshold, and any extra
    suppression needed to avoid several detections on one cell.
    """
    response = np.asarray(response, dtype=np.float64)
    local_max = response == ndimage.maximum_filter(
        response, size=neighborhood_size, mode="reflect"
    )
    threshold = threshold_rel * response.max()
    candidates = local_max & (response > threshold) & (response > 0)

    # A flat-topped peak gives several equal maxima; keep one pixel per
    # 8-connected plateau.
    labels, n = ndimage.label(candidates, structure=np.ones((3, 3)))
    maxima = np.zeros(response.shape, dtype=bool)
    if n == 0:
        return maxima
    peaks = ndimage.maximum_position(response, labels, index=np.arange(1, n + 1))
    peaks = sorted(peaks, key=lambda p: response[p], reverse=True)

    # Greedy non-maximum suppression: a weaker peak closer than half the
    # neighborhood to a stronger one is treated as the same cell.
    min_dist = neighborhood_size / 2.0
    kept = []
    for r, c in peaks:
        if all((r - kr) ** 2 + (c - kc) ** 2 >= min_dist ** 2 for kr, kc in kept):
            kept.append((r, c))
    for r, c in kept:
        maxima[r, c] = True
    return maxima


def morphology_mask(
    image,
    threshold=0.28,
    erosion_iterations=1,
    dilation_iterations=2,
    closing_iterations=1,
    background_sigma=0,
):
    """Segment cells and clean the mask with erosion and dilation.

    TODO:
      1. Threshold the image to obtain a binary foreground mask.
      2. Use binary erosion and binary dilation to perform opening and/or
         closing. Explain your operation order and structuring element.
      3. Return the cleaned boolean mask.

    The parameters above are only starting points; tune them per image.

    Scalar image: if ``background_sigma`` > 0, a wide Gaussian blur of the
    image (the slowly varying illumination and out-of-focus haze) is
    subtracted first, so one threshold works across the whole field.

    Operation order (structuring element: 3x3 cross, iterated):
      a. Closing (dilate, then erode, ``closing_iterations`` times each) fills
         pinholes and dark gaps inside a cell before anything is removed.
      b. Erosion (``erosion_iterations``) deletes isolated noise pixels and
         cuts the thin bridges where two cells only touch weakly.
      c. Dilation (``dilation_iterations``) grows the surviving cores back.
    With equal iteration counts, steps b and c form an opening. Using fewer
    dilations than erosions keeps the cores of touching cells apart.
    """
    image = np.asarray(image, dtype=np.float64)
    if background_sigma > 0:
        image = image - gaussian_filter(image, background_sigma)
    mask = image > threshold
    if closing_iterations > 0:
        mask = ndimage.binary_dilation(mask, STRUCTURE, iterations=closing_iterations)
        mask = ndimage.binary_erosion(
            mask, STRUCTURE, iterations=closing_iterations, border_value=1
        )
    if erosion_iterations > 0:
        mask = ndimage.binary_erosion(mask, STRUCTURE, iterations=erosion_iterations)
    if dilation_iterations > 0:
        mask = ndimage.binary_dilation(mask, STRUCTURE, iterations=dilation_iterations)
    return mask


def count_components(mask, min_area=12, max_area=400):
    """Count connected foreground components in an area range.

    TODO: label connected components with ``scipy.ndimage.label``. Measure
    their areas, reject components outside [min_area, max_area], and return
    ``(count, labels, kept_label_ids)``.
    """
    labels, n = ndimage.label(mask, structure=STRUCTURE)
    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    ids = np.arange(1, n + 1)
    kept_label_ids = ids[(areas >= min_area) & (areas <= max_area)]
    return len(kept_label_ids), labels, kept_label_ids


def save_mask(mask, path):
    """Save a boolean mask as a black-and-white PNG."""
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_response(response, path):
    """Normalize a floating-point response for visualization."""
    response = np.asarray(response, dtype=np.float32)
    lo, hi = float(response.min()), float(response.max())
    scaled = np.zeros_like(response) if hi <= lo else (response - lo) / (hi - lo)
    Image.fromarray((scaled * 255).astype(np.uint8), mode="L").save(path)


def save_blob_overlay(source_path, maxima, radius, output_path):
    """Draw a circle around every detected local maximum."""
    canvas = Image.open(source_path).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    rows, cols = np.nonzero(maxima)
    for row, col in zip(rows, cols):
        draw.ellipse(
            (col - radius, row - radius, col + radius, row + radius),
            outline=(255, 255, 0),
            width=1,
        )
    canvas.save(output_path)


def save_component_overlay(source_path, labels, kept_label_ids, output_path):
    """Draw bounding boxes and component numbers for retained cells."""
    canvas = Image.open(source_path).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    objects = ndimage.find_objects(labels)
    kept = set(int(label_id) for label_id in kept_label_ids)
    count = 0
    for label_id, slices in enumerate(objects, start=1):
        if label_id not in kept or slices is None:
            continue
        count += 1
        row_slice, col_slice = slices
        box = (
            col_slice.start,
            row_slice.start,
            col_slice.stop - 1,
            row_slice.stop - 1,
        )
        draw.rectangle(box, outline=(255, 64, 64), width=1)
        draw.text((box[0], box[1]), str(count), fill=(255, 255, 0))
    canvas.save(output_path)


def process_cell(path, parameters):
    """Run the morphology pipeline and save report-ready diagnostics."""
    image = read_gray(path)
    mask = morphology_mask(
        image,
        threshold=parameters["threshold"],
        erosion_iterations=parameters["erosion_iterations"],
        dilation_iterations=parameters["dilation_iterations"],
        closing_iterations=parameters.get("closing_iterations", 1),
        background_sigma=parameters.get("background_sigma", 0),
    )
    count, labels, kept_ids = count_components(
        mask,
        min_area=parameters["min_area"],
        max_area=parameters["max_area"],
    )
    stem = path.stem
    raw_mask = morphology_mask(
        image, parameters["threshold"], 0, 0, 0, parameters.get("background_sigma", 0)
    )
    save_mask(raw_mask, OUTPUT_DIR / f"{stem}_threshold.png")
    save_mask(mask, OUTPUT_DIR / f"{stem}_mask.png")
    save_component_overlay(
        path,
        labels,
        kept_ids,
        OUTPUT_DIR / f"{stem}_morphology.png",
    )
    return count


def detect_blobs(path, sigma_small, sigma_large, neighborhood_size, threshold_rel, tag=""):
    """Run DoG blob detection on one image, save the response and overlay, return the count."""
    response = difference_of_gaussians(read_gray(path), sigma_small, sigma_large)
    maxima = find_blob_maxima(
        response, neighborhood_size=neighborhood_size, threshold_rel=threshold_rel
    )
    save_response(response, OUTPUT_DIR / f"{path.stem}_dog{tag}.png")
    save_blob_overlay(
        path,
        maxima,
        radius=max(2, round(np.sqrt(2) * sigma_small)),
        output_path=OUTPUT_DIR / f"{path.stem}_blobs{tag}.png",
    )
    return int(maxima.sum())


SELECTED = ["002cell.png", "050cell.png", "100cell.png", "150cell.png"]
DEFAULT_PARAMETERS = {
    "background_sigma": 10,
    "threshold": 0.12,
    "closing_iterations": 1,
    "erosion_iterations": 3,
    "dilation_iterations": 2,
    "min_area": 15,
    "max_area": 400,
}
# Per-image settings used in the report.
IMAGE_PARAMETERS = {
    "002cell.png": dict(DEFAULT_PARAMETERS),
    "050cell.png": dict(DEFAULT_PARAMETERS),
    "100cell.png": dict(DEFAULT_PARAMETERS, threshold=0.15),
    "150cell.png": dict(DEFAULT_PARAMETERS),
}
# Starter settings without background flattening, kept as a baseline.
BASELINE_PARAMETERS = {
    "background_sigma": 0,
    "threshold": 0.28,
    "closing_iterations": 1,
    "erosion_iterations": 1,
    "dilation_iterations": 1,
    "min_area": 12,
    "max_area": 400,
}

# Task 5(a) settings on SELECTED[0].
SCALE_PAIRS = [(1.5, 2.4), (2.0, 3.2), (2.5, 4.0), (3.0, 4.8), (3.5, 5.6)]
DOG_SIGMAS = (2.5, 4.0)
DOG_NEIGHBORHOOD = 9
DOG_THRESHOLD_REL = 0.30


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = SELECTED
    image_parameters = IMAGE_PARAMETERS

    print("Task 5(b): morphology cell counting")
    morphology_counts = {}
    for filename in selected:
        path = CELL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(path)
        parameters = image_parameters[filename]
        count = process_cell(path, parameters)
        morphology_counts[filename] = count
        print(f"{filename}: {count} cells  {parameters}")

    # Task 5(a): Blob-detection experiment for one selected image.
    blob_path = CELL_DIR / selected[0]
    neighborhood_size, threshold_rel = DOG_NEIGHBORHOOD, DOG_THRESHOLD_REL
    print(f"Task 5(a): DoG blob detection on {blob_path.name}")
    for sigma_small, sigma_large in SCALE_PAIRS:
        tag = f"_s{sigma_small:g}_{sigma_large:g}"
        n = detect_blobs(blob_path, sigma_small, sigma_large,
                         neighborhood_size, threshold_rel, tag=tag)
        print(f"  sigma = ({sigma_small}, {sigma_large}), neighborhood = "
              f"{neighborhood_size}, threshold_rel = {threshold_rel}: {n} DoG maxima")

    sigma_small, sigma_large = DOG_SIGMAS
    blob_count = detect_blobs(blob_path, sigma_small, sigma_large,
                              neighborhood_size, threshold_rel)
    print(f"{blob_path.name}: {blob_count} DoG maxima "
          f"(selected sigma = ({sigma_small}, {sigma_large}))")

    # Task 5(c): compare the two counts and vary one parameter of each method.
    print(f"Task 5(c): {blob_path.name}: DoG maxima = {blob_count}, "
          f"morphology components = {morphology_counts[blob_path.name]}")
    response = difference_of_gaussians(read_gray(blob_path), sigma_small, sigma_large)
    for rel in (0.1, 0.2, 0.3, 0.4, 0.5):
        n = int(find_blob_maxima(response, neighborhood_size, rel).sum())
        print(f"  DoG threshold_rel = {rel}: {n} maxima")
    for nb in (5, 7, 9, 11, 13):
        n = int(find_blob_maxima(response, nb, threshold_rel).sum())
        print(f"  DoG neighborhood_size = {nb}: {n} maxima")

    image = read_gray(blob_path)
    base = image_parameters[blob_path.name]

    def morphology_count(**changes):
        p = dict(base, **changes)
        mask = morphology_mask(image, p["threshold"], p["erosion_iterations"],
                               p["dilation_iterations"], p["closing_iterations"],
                               p["background_sigma"])
        return count_components(mask, p["min_area"], p["max_area"])[0]

    for t in (0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21):
        print(f"  morphology threshold = {t}: {morphology_count(threshold=t)} components")
    for e in (1, 2, 3, 4):
        print(f"  morphology erosion = {e} (dilation 2): "
              f"{morphology_count(erosion_iterations=e)} components")
    print(f"  morphology baseline without flattening {BASELINE_PARAMETERS}: "
          f"{morphology_count(**BASELINE_PARAMETERS)} components")


if __name__ == "__main__":
    main()
