from matplotlib import pyplot
from skimage import io
import numpy as np
from GPU import CL
import cv2


class ImageFilters:
    EdgeDetectionKernels = 'kernels.cl'

    def __init__(self, image: np.ndarray):
        self.GPU_process = CL()
        self.original = image

    def detect_edges(self, blur_ratio=5, low_threshold_ratio=0.05, high_threshold_ratio=0.09, weak=np.uint32(25),
                     strong=np.uint32(255)):
        self.GPU_process.load_program(ImageFilters.EdgeDetectionKernels)

        result = self.normalize(self.original)
        result = self.rgb2gray(result).astype(np.uint8)
        result = cv2.blur(result, (blur_ratio, blur_ratio))

        self.GPU_process.load_image(np.ascontiguousarray(result))
        result, angle_matrix = self.GPU_process.execute('GradientCalculation')
        angle_matrix[angle_matrix < 0] += 180

        self.GPU_process.load_image(np.ascontiguousarray(result))
        result = self.GPU_process.execute('NonMaxSuppression', angle_matrix)

        result = self.threshold(result, low_threshold_ratio, high_threshold_ratio, weak, strong)

        self.GPU_process.load_image(np.ascontiguousarray(result))
        result = self.GPU_process.execute('hysteresis', weak, strong)

        return result

    @staticmethod
    def reduce_noise(image: np.ndarray):
        cv2.blur(image, (5, 5))

    @staticmethod
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

    @staticmethod
    def threshold(img, low_threshold_ratio, high_threshold_ratio, weak: np.uint32, strong: np.uint32):
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

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


if __name__ == '__main__':
    # example
    original = io.imread("images/dog.jpg")
    filters = ImageFilters(original)
    edges = filters.detect_edges()

    io.imshow(edges, cmap=pyplot.get_cmap('gray'))
    io.show()
