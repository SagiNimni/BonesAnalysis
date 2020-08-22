from skimage import io
from matplotlib import pyplot
import numpy as np
from GPU import CL
import cv2
import time


class ImageFilters:
    EdgeDetectionKernels = 'kernels.cl'

    def __init__(self, image: np.ndarray):
        self.GPU_process = CL()
        self.original = image

    def detect_edges(self, blur_ratio=5, low_threshold_ratio=0.05, high_threshold_ratio=0.09, weak=np.uint32(25),
                     strong=np.uint32(255)):
        start_time = time.time()

        self.GPU_process.load_program(ImageFilters.EdgeDetectionKernels)
        print("3%- C Program Built")

        result = self.normalize(self.original)
        print("7%- normalized")

        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        print("10%- Grayscaled")

        io.imsave('result.tif', result)
        cv = cv2.cvtColor(cv2.imread('result.tif').astype(np.uint8), cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(cv, cv2.HOUGH_GRADIENT, dp=40, minDist=5000, minRadius=1500, maxRadius=3000)
        self.a = circles[0, 0, 0]
        self.b = circles[0, 0, 1]
        self.r = circles[0, 0, 2]

        result = cv2.blur(result, (blur_ratio, blur_ratio))
        print("15%- Noise Reduction")

        self.GPU_process.load_image(result)
        result, angle_matrix = self.GPU_process.execute('GradientCalculation', self.a, self.b, self.r)
        angle_matrix[angle_matrix < 0] += 180
        print("35%- Gradient Calculation")

        self.GPU_process.load_image(result)
        result = self.GPU_process.execute('NonMaxSuppression', angle_matrix)
        print("60%- Non Maximum Suppression")

        result = self.threshold(result, low_threshold_ratio, high_threshold_ratio, weak, strong)
        print("80%- Double Threshold")

        self.GPU_process.load_image(result)
        result = self.GPU_process.execute('hysteresis', weak, strong)
        print("100%- Edge Tracking by Hysteresis \nEdge Detection Filter Successfully Completed!")

        return result

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
        low_threshold = img.max() * low_threshold_ratio

        M, N = img.shape
        res = np.zeros((M, N), dtype=np.uint8)

        strong_i, strong_j = np.where(img >= high_threshold)
        zeros_i, zeros_j = np.where(img < low_threshold)
        weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res
