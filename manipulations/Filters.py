from skimage import io
import numpy as np
from manipulations.ParralerMethods.GPU import CL
import cv2
from disjoint_set import DisjointSet


class ImageFilters:
    EdgeDetectionKernels = 'edgeFilters.cl'
    FindObjectsKernels = 'objectFilters.cl'

    def __init__(self, image: np.ndarray):
        self.GPU_process = CL()
        self.original = image
        self.edge_image = None
        self.labels = None

    def detect_edges(self, blur_ratio=5, low_threshold_ratio=0.05, high_threshold_ratio=0.09, weak=np.uint32(25),
                     strong=np.uint32(255)) -> np.ndarray:

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

        self.edge_image = result
        return result

    def connected_components(self) -> tuple:
        if self.edge_image is None:
            self.detect_edges()
        unionfind = DisjointSet()

        # First Pass
        label = 1
        labels = np.zeros_like(self.edge_image).astype(np.uint32)
        for i in range(1, self.edge_image.shape[0]-1):
            for k in range(1, self.edge_image.shape[1]-1):
                pix = self.edge_image[i, k]
                if pix == 255:
                    neighbours = np.unique(np.array([labels[i, k-1], labels[i-1, k-1], labels[i-1, k],
                                           labels[i-1, k+1]]))
                    if neighbours[:].any() != 0:
                        neighbours = np.delete(neighbours, np.where(neighbours == 0))
                        labels[i, k] = neighbours[0]
                        for n in range(1, neighbours.shape[0]):
                            unionfind.union(neighbours[n], neighbours[0])

                    else:
                        labels[i, k] = label
                        unionfind.union(label, label)
                        label += 1

        # Second Pass
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
        self.labels = (unique_labels, labels)
        print("All The Components Was Successfully Connected!")
        return self.labels

    def remove_small_objects(self, low_threshold=50):
        if self.labels is None:
            self.connected_components()
        self.GPU_process.load_program(ImageFilters.FindObjectsKernels)
        print("30%- C Program Built")

        self.GPU_process.load_image(self.labels[1])
        result = self.GPU_process.execute('removeSmallEdges', self.labels[0], low_threshold)
        print("100%- Removes small objects")

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
