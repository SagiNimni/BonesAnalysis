from skimage import io
import numpy as np
from matplotlib import pyplot
from GPU import CL
import cv2


def normalize(image: np.ndarray):
    red = image[:, :, 0]
    norm_red = (red - red.min()) * 255.0 / (red.max() - red.min())

    green = image[:, :, 1]
    norm_green = (green - green.min()) * 255.0 / (green.max() - green.min())

    blue = image[:, :, 2]
    norm_blue = (blue - blue.min()) * 255.0 / (blue.max() - blue.min())

    norm = image
    norm[:, :, 0] = norm_red
    norm[:, :, 1] = norm_green
    norm[:, :, 2] = norm_blue

    return norm


def run_gpu_kernel(kernel, kernel_method: str, image: np.ndarray, *args):
    norm = np.ascontiguousarray(image)
    process_gpu = CL()
    process_gpu.load_program(kernel)
    process_gpu.load_image(norm)
    return process_gpu.execute(kernel_method, *args)


def threshold(img, low_threshold_ratio=0.05, high_threshold_ratio=0.09, weak=np.uint32(25), strong=np.uint32(255)):
    high_threshold = img.max() * high_threshold_ratio
    low_threshold = high_threshold_ratio * low_threshold_ratio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    strong_i, strong_j = np.where(img >= high_threshold)
    zeros_i, zeros_j = np.where(img < low_threshold)
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def main():
    original = io.imread("images/4_333.tif")
    norm = normalize(original)
    grayed = rgb2gray(norm).astype(np.uint8)
    blurred = cv2.blur(grayed, (5, 5))

    result, angle_matrix = run_gpu_kernel('kernels.cl', 'GradientCalculation', blurred)
    angle_matrix[angle_matrix < 0] += 180
    result = run_gpu_kernel('kernels.cl', 'NonMaxSuppression', result, angle_matrix)
    result = threshold(result, high_threshold_ratio=0.07)
    result = run_gpu_kernel('kernels.cl', 'hysteresis', result, 25, 255)

    io.imshow(result, cmap=pyplot.get_cmap('gray'), vmin=0, vmax=255)
    io.show()


if __name__ == '__main__':
    main()
