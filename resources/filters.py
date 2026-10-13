import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import read_img, save_img


def image_patches(image, patch_size=(16, 16)):
    """
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.

    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """
    # TODO: Use slicing to complete the function
    M, N = patch_size
    H, W = image.shape
    output = []
    for i in range(0, H - M + 1, M):
        for j in range(0, W - N + 1, N):
            output.append(image[i:i + M, j:j + N])
    return output


def gaussian_kernel(size, sigma, normalize=False):
    """
    Return a size x size 2D Gaussian kernel
    G(x, y) = 1 / (2 pi sigma^2) * exp(-(x^2 + y^2) / (2 sigma^2)),
    where (0, 0) is the center of the kernel.
    If normalize is True, the kernel is divided by its sum.
    """
    x = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(x, x)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    if normalize:
        kernel = kernel / kernel.sum()
    return kernel


def gaussian_kernel_1d(size, sigma):
    """Return a 1D Gaussian G(x) = 1 / sqrt(2 pi sigma^2) * exp(-x^2 / (2 sigma^2)) as a 1 x size array."""
    x = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-x ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)
    return g[None, :]


def convolve(image, kernel):
    """
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if kernel.ndim == 1:
        kernel = kernel[None, :]
    # Flip the kernel so that the sliding-window sum below is a convolution.
    kernel = kernel[::-1, ::-1]

    H, W = image.shape
    h, w = kernel.shape
    # The kernel center is (h // 2, w // 2), the same convention as
    # scipy.ndimage.convolve, so even-sized kernels also match scipy.
    pad_top, pad_bottom = (h - 1) // 2, h // 2
    pad_left, pad_right = (w - 1) // 2, w // 2
    padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='constant', constant_values=0)

    output = np.zeros((H, W), dtype=float)
    for i in range(h):
        for j in range(w):
            output += kernel[i, j] * padded[i:i + H, j:j + W]
    return output


def edge_detection(image):
    """
    Return Ix, Iy and the gradient magnitude of the input image

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # TODO: Fix kx, ky
    # Ix(x, y) = I(x + 1, y) - I(x - 1, y), Iy(x, y) = I(x, y + 1) - I(x, y - 1)
    kx = np.array([[1, 0, -1]])  # 1 x 3
    ky = np.array([[1], [0], [-1]])  # 3 x 1

    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt(Ix ** 2 + Iy ** 2)

    return Ix, Iy, grad_magnitude


def sobel_operator(image):
    """
    Return Gx, Gy, and the gradient magnitude.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    # TODO: Use convolve() to complete the function
    Sx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    Sy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    Gx = convolve(image, Sx)
    Gy = convolve(image, Sy)
    grad_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

    return Gx, Gy, grad_magnitude


def save_gray_plot(img, path, title=None, vmin=None, vmax=None, cmap='gray', colorbar=True):
    """Save an image with matplotlib so its intensity scale stays visible."""
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.axis('off')
    if colorbar:
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(path, "is saved!")


def intensity_stats(name, img):
    print("%s: min = %.4f, max = %.4f, range = %.4f, mean = %.4f"
          % (name, img.min(), img.max(), img.max() - img.min(), img.mean()))


def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- Image Patches (warm-up, not graded in Assignment2.pdf) --
    # First complete image_patches()
    patches = image_patches(img)
    # Now choose any three patches and save them
    # chosen_patches should have those patches stacked vertically/horizontally
    chosen_patches = np.hstack([patches[0], patches[len(patches) // 2], patches[-1]])
    save_img(chosen_patches, "./image_patches/q1_patch.png")

    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # -- TODO Task 1: Convolution and Gaussian Filter --
    # (a): No code. Numerical check of the separability shown in the report.
    G = gaussian_kernel(3, 0.572)
    Gx = gaussian_kernel_1d(3, 0.572)  # 1 x 3, horizontal
    Gy = Gx.T  # 3 x 1, vertical
    print("Task 1(a): max |G - Gy Gx| = %.3e" % np.abs(G - Gy @ Gx).max())
    separable = convolve(convolve(img, Gy), Gx)
    print("Task 1(a): max |I*G - (I*Gy)*Gx| = %.3e"
          % np.abs(convolve(img, G) - separable).max())

    # (b): Complete convolve()

    # (c)
    # Calculate the Gaussian kernel described in the question.
    # There is tolerance for the kernel.
    kernel_gaussian = gaussian_kernel(3, 0.572)
    print("Task 1(c): 3x3 Gaussian kernel, sigma = 0.572\n", np.round(kernel_gaussian, 4))

    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")
    save_gray_plot(img, "./gaussian_filter/q2_original_plot.png",
                   "Original", vmin=0, vmax=255)
    save_gray_plot(filtered_gaussian, "./gaussian_filter/q2_gaussian_plot.png",
                   "Gaussian 3x3, sigma = 0.572", vmin=0, vmax=255)

    # (d)
    kernel_norm = gaussian_kernel(3, 0.572, normalize=True)
    kernel_sigma2 = gaussian_kernel(3, 2.0)
    print("Task 1(d): sum of 3x3 kernel, sigma = 0.572: %.6f" % kernel_gaussian.sum())
    print("Task 1(d): sum of 3x3 kernel, sigma = 2 (not normalized): %.6f" % kernel_sigma2.sum())
    filtered_norm = convolve(img, kernel_norm)
    filtered_sigma2 = convolve(img, kernel_sigma2)
    intensity_stats("Task 1(d) original", img)
    intensity_stats("Task 1(d) sigma = 0.572 (raw)", filtered_gaussian)
    intensity_stats("Task 1(d) sigma = 0.572 (sum = 1)", filtered_norm)
    intensity_stats("Task 1(d) sigma = 2 (not normalized)", filtered_sigma2)
    save_gray_plot(filtered_norm, "./gaussian_filter/q2d_sigma0572_norm.png",
                   "sigma = 0.572, sum = 1", vmin=0, vmax=255)
    save_gray_plot(filtered_sigma2, "./gaussian_filter/q2d_sigma2_raw.png",
                   "sigma = 2, sum = %.3f" % kernel_sigma2.sum(), vmin=0, vmax=255)

    # (e): No code

    # (f): Complete edge_detection()

    # (g)
    # Use edge_detection() to detect edges
    # for the orignal and gaussian filtered images.
    _, _, edge_detect = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")
    vmax_edge = edge_detect.max()
    save_gray_plot(edge_detect, "./gaussian_filter/q3_edge_plot.png",
                   "Gradient magnitude: original", vmin=0, vmax=vmax_edge)
    save_gray_plot(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian_plot.png",
                   "Gradient magnitude: Gaussian filtered", vmin=0, vmax=vmax_edge)
    intensity_stats("Task 1(g) edge, original", edge_detect)
    intensity_stats("Task 1(g) edge, Gaussian filtered", edge_with_gaussian)

    print("Gaussian Filter is done. ")

    # -- TODO Task 2: Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # (a): No code. Numerical comparison of I * Sx and (I * GS) * kx.
    GS = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    kx = np.array([[1, 0, -1]])
    print("Task 2(a): GS * kx =\n", convolve(np.pad(GS, ((0, 0), (1, 1))), kx).astype(int))
    smoothed_derivative = convolve(convolve(img, GS), kx)

    # (b): Complete sobel_operator()

    # (c)
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")
    corr = np.corrcoef(Gx.ravel(), smoothed_derivative.ravel())[0, 1]
    print("Task 2(a): correlation between I*Sx and (I*GS)*kx = %.4f" % corr)
    save_gray_plot(Gx, "./sobel_operator/q2_Gx_plot.png", "I * Sx", cmap='gray')
    save_gray_plot(Gy, "./sobel_operator/q2_Gy_plot.png", "I * Sy", cmap='gray')
    save_gray_plot(edge_sobel, "./sobel_operator/q2_edge_sobel_plot.png",
                   "Sobel gradient magnitude", cmap='gray')

    print("Sobel Operator is done. ")

    # -- TODO Task 3: LoG Filter --
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # (a)
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])
    # Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    filtered_LoG1 = convolve(img, kernel_LoG1)
    filtered_LoG2 = convolve(img, kernel_LoG2)
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")
    save_img(np.abs(filtered_LoG1), "./log_filter/q1_LoG1_abs.png")
    save_img(np.abs(filtered_LoG2), "./log_filter/q1_LoG2_abs.png")
    print("Task 3(a): kernel sums: LoG1 = %d, LoG2 = %d"
          % (kernel_LoG1.sum(), kernel_LoG2.sum()))
    intensity_stats("Task 3(a) LoG1 response", filtered_LoG1)
    intensity_stats("Task 3(a) LoG2 response", filtered_LoG2)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, kernel, name in ((axs[0], kernel_LoG1, "kernel_LoG1 (3x3)"),
                             (axs[1], kernel_LoG2, "kernel_LoG2 (9x9)")):
        im = ax.imshow(kernel, cmap='coolwarm')
        ax.set_title(name)
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("./log_filter/q1_LoG_kernels.png", dpi=150)
    plt.close()

    # (b)
    # Follow instructions in pdf to approximate LoG with a DoG
    data = np.load('log1d.npz')
    log50 = data['log50']
    gauss50 = data['gauss50']
    gauss53 = data['gauss53']
    dog = gauss53 - gauss50
    # Least-squares scale so that the DoG and the LoG can be drawn on one axis.
    scale = np.dot(dog, log50) / np.dot(dog, dog)
    rel_err = np.linalg.norm(scale * dog - log50) / np.linalg.norm(log50)
    print("Task 3(b): best scale = %.4f, relative L2 error of scaled DoG vs LoG = %.4f"
          % (scale, rel_err))
    print("Task 3(b): sums: log50 = %.3e, gauss50 = %.4f, gauss53 = %.4f, DoG = %.3e"
          % (log50.sum(), gauss50.sum(), gauss53.sum(), dog.sum()))

    x = np.arange(len(log50)) - len(log50) // 2
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(x, gauss50, label='Gaussian, sigma = 50')
    axs[0].plot(x, gauss53, label='Gaussian, sigma = 53')
    axs[0].set_title('Two Gaussians')
    axs[0].legend()
    axs[1].plot(x, dog, color='C2', label='gauss53 - gauss50 (DoG)')
    axs[1].set_title('Difference of Gaussians')
    axs[1].legend()
    axs[2].plot(x, log50, color='C3', label='LoG, sigma = 50')
    axs[2].plot(x, scale * dog, '--', color='k', label='DoG x %.1f' % scale)
    axs[2].set_title('LoG vs scaled DoG')
    axs[2].legend()
    for ax in axs:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./log_filter/q2_dog_vs_log.png", dpi=150)
    plt.close()
    print("./log_filter/q2_dog_vs_log.png is saved!")

    print("LoG Filter is done. ")


if __name__ == "__main__":
    main()
