from manipulations.ParralerMethods.GPU import CL
from disjoint_set import DisjointSet
from skimage import io, morphology
import numpy as np
import math
import cv2
import os
import sys


class ImageFilters:
    EdgeDetectionKernels = 'edgeFilters.cl'
    FindObjectsKernels = 'objectFilters.cl'
    ColorKernels = 'colorFilters.cl'

    def __init__(self):
        self.GPU_process = CL()
        self.current_program = ''

    def canny(self, image: np.ndarray, mask=None, gradient_ratio=2, blur_ratio=5, low_threshold_ratio=0.05,
              high_threshold_ratio=0.09, weak=np.uint32(25), strong=np.uint32(255), mute=True) -> np.ndarray:
        """
        This method uses the GPU cumputing power for some functions.
        Canny edge detection is an algrotim to discover edges in images and it includes four steps:
        1. Gradient Calculation - find the edges
        2. Non-Maximum Suppression - thin the edges
        3. Double Threshold - filter weak edges
        4. Hysteresis - final filter of weak edges and paint every pixel as black or white

        prerequisites for the image:
        1. grayscaled
        2. normalized

        :param mute: mute the console output
        :param gradient_ratio: The intensity of edge detection gradient
        :param blur_ratio: the ratio for noise reduction
        :param image: The image to process
        :param low_threshold_ratio: threshold for double threshold step
        :param high_threshold_ratio: threshold for double threshold step
        :param weak: color for double threshold step
        :param strong: color for double threshold step
        :param mask: places that should not be included in the edge detection filter
        :return: A image matrix of edges
        """
        if mask is None:
            mask = np.zeros_like(image)

        if mute:
            print("Canny Filter...", end='', flush=True)
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        if self.current_program != ImageFilters.EdgeDetectionKernels:
            self.current_program = ImageFilters.EdgeDetectionKernels
            self.GPU_process.load_program(self.current_program)

        print("\nCanny Filter...")
        result = self._noise_reduction_(image, ratio=blur_ratio)
        print("15%- Noise Reduction")

        io.imsave('result.tif', result)
        cv = cv2.cvtColor(cv2.imread('result.tif').astype(np.uint8), cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(cv, cv2.HOUGH_GRADIENT, dp=40, minDist=5000, minRadius=1500, maxRadius=3000)
        try:
            a = circles[0, 0, 0]
            b = circles[0, 0, 1]
            r = circles[0, 0, 2]
        except TypeError:
            a = 0
            b = 0
            r = math.sqrt((image.shape[0]/2)**2 + (image.shape[1]/2)**2) * 2
        os.remove('result.tif')

        self.GPU_process.load_image(result)
        result, angle_matrix = self.GPU_process.execute('GradientCalculation', mask, gradient_ratio, a, b, r)
        angle_matrix[angle_matrix < 0] += 180
        print("35%- Gradient Calculation")

        self.GPU_process.load_image(result)
        result = self.GPU_process.execute('NonMaxSuppression', angle_matrix)
        print("60%- Non Maximum Suppression")

        result = self._threshold_(result, low_threshold_ratio, high_threshold_ratio, weak, strong)
        print("80%- Double Threshold")

        self.GPU_process.load_image(result)
        result = self.GPU_process.execute('hysteresis', weak, strong)
        print("100%- Edge Tracking by Hysteresis \nCanny Edge Detection Successfully Completed!")

        if mute:
            sys.stdout = old_stdout
            print("[DONE]")

        return result

    def remove_small_components(self, labels: tuple, low_threshold=50, mute=True):
        """
        This method uses the labels result from the "connected_components" method to find components
        that are smaller than the threshold and remove them.

        :param mute: mute the console output
        :param labels: Result from the "connected_components" method.
        :param low_threshold: The smallest amount of conneted pixels a component can have.
        :return: A matrix of the image without the small objects.
        """
        if mute:
            print("Small Objects Noise Reduction...", end='', flush=True)
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        if self.current_program != ImageFilters.FindObjectsKernels:
            self.current_program = ImageFilters.FindObjectsKernels
            self.GPU_process.load_program(self.current_program)

        print("\nSmall Objects Noise Reduction...")
        print("30%- C Program Built")

        self.GPU_process.load_image(labels[1])
        result = self.GPU_process.execute('removeSmallEdges', labels[0], low_threshold)
        print("100%- Removed small objects")
        print("Clear Small Objects Filter Was Successfully Completed!")

        if mute:
            sys.stdout = old_stdout
            print("[DONE]")

        return result

    def make_mask_background(self, image: np.ndarray, low_mask: tuple, high_mask: tuple):
        print("Remove Mask...", end='', flush=True)

        if self.current_program != ImageFilters.ColorKernels:
            self.current_program = ImageFilters.ColorKernels
            self.GPU_process.load_program(self.current_program)

        self.GPU_process.load_image(image.astype(np.uint8))
        result = self.GPU_process.execute('makeBackground', low_mask, high_mask)

        print("[DONE]")

        return result

    def remove_shapes_inside_shape(self, image: np.ndarray, shape_size=10):
        print("Removes Shapes Inside Shape...", end='', flush=True)

        if self.current_program != ImageFilters.FindObjectsKernels:
            self.current_program = ImageFilters.FindObjectsKernels
            self.GPU_process.load_program(self.current_program)

        self.GPU_process.load_image(image.astype(np.uint8))
        result = self.GPU_process.execute('removeShapesInsideShape', shape_size)
        result[np.where(result == 70)] = 0

        print("[DONE]")
        return result

    @staticmethod
    def fill_holes(image: np.ndarray, filling_range=400):
        print("Hole Filling...", end='', flush=True)
        result = morphology.area_closing(image, area_threshold=filling_range)
        print('[DONE]')
        return result

    @staticmethod
    def binary_dilation(image: np.ndarray, neighborhood=3):
        print("dilation...", end='', flush=True)
        selem = np.ones((neighborhood, neighborhood), dtype=np.uint8)
        result = morphology.binary_dilation(image, selem=selem)
        print('[DONE]')
        return result

    @staticmethod
    def binary_erosion(image: np.ndarray, neighborhood=3):
        print("erosion...", end='', flush=True)
        selem = np.ones((neighborhood, neighborhood), dtype=np.uint8)
        result = morphology.binary_erosion(image, selem=selem)
        print('[DONE]')
        return result

    @staticmethod
    def connected_components(image: np.ndarray, mute=True) -> tuple:
        """
        This method uses the Two Pass algorithm to find all connected components inside an image.
        Connected components is defined as foreground pixels(255) that are connected in the image matrix.
        The Two Pass algorithm consists of two steps:
        1. Looping through all the matrix and put connected components in the unionfind
        2. Compress the unionfind the be more efficient.

       prerequisites:
       1. normalized
       2. binary (black or white values)

        :param mute: mute the console output
        :param image: The image for processing
        :return: Labels matrix that contains all connected components and a list of the connected components
        """
        if mute:
            print("Connected Components Labeling...", end='', flush=True)
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        print("\nConnected Components Labeling...")
        unionfind = DisjointSet()

        print("First Pass")
        label = 1
        labels = np.zeros_like(image).astype(np.uint32)
        for i in range(1, image.shape[0] - 1):
            for k in range(1, image.shape[1] - 1):
                pix = image[i, k]
                if pix == 255:
                    neighbours = np.unique(np.array([labels[i, k - 1], labels[i - 1, k - 1], labels[i - 1, k],
                                                     labels[i - 1, k + 1]]))
                    if neighbours[:].any() != 0:
                        neighbours = np.delete(neighbours, np.where(neighbours == 0))
                        labels[i, k] = neighbours[0]
                        for n in range(1, neighbours.shape[0]):
                            unionfind.union(neighbours[n], neighbours[0])

                    else:
                        labels[i, k] = label
                        unionfind.union(label, label)
                        label += 1

        print("Second Pass")
        i = 0
        for row in labels:
            j = 0
            for pix in row:
                if pix != 0:
                    root = unionfind.find(pix)
                    if root != pix:
                        labels[i, j] = root
                j += 1
            i += 1

        unique, counts = np.unique(labels, return_counts=True)
        unique_labels = dict(zip(unique, counts))
        print("Connected Components Labeling Filter Was Successfully Completed!\n")

        if mute:
            sys.stdout = old_stdout
            print("[DONE]")

        return unique_labels, labels

    @staticmethod
    def grayscale(image: np.ndarray):
        """
        Use cv2 methods to turn the image from RGB to GrayScale with dtype=numpy.uint8

        :param image: A RGB image to grayscale (numpy.ndarray)
        :return: A grayscaled matrix of the image (numpy.ndarray)
        """
        print("Grayscale...", end='')
        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        print("[DONE]")
        return result

    @staticmethod
    def normalize(image: np.ndarray):
        """
        Normilization of image's matrix to RGB format 0-255 from any other range

        :param image: The image to normalize (numpy.ndarray)
        :return: The normalized image's matrix (numpy.ndarray)
        """
        print("normalization...", end='')
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

        print("[DONE]")
        return norm

    @staticmethod
    def _threshold_(img: np.ndarray, low_threshold_ratio, high_threshold_ratio, weak: np.uint32, strong: np.uint32):
        """
        Private method for canny edge detection third step "Double Threshold"

        :param img: The image to process (numpy.ndarray)
        :param low_threshold_ratio:  The threshold for low pixels to be recognized as edge
        :param high_threshold_ratio: The threshold for high pixels to be recognized as edge
        :param weak: The weak color for weak edges (np.uint32)
        :param strong: The strong color for strong edges (numpy.uint32)
        :return: The result matrix after double threshold (numpy.ndarray)
        """
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

    @staticmethod
    def _noise_reduction_(image: np.ndarray, ratio):
        """
        Using the CV2 library blur function to reduce noise in the image

        :param image: The image to blur (numpy.ndarray)
        :param blur_ratio: The bigger the number the harder the blur
        :return: A blurred matrix of the image
        """
        return cv2.GaussianBlur(image, (ratio, ratio), 0)
