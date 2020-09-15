from manipulations.Filters import ImageFilters
from skimage import io, morphology
import numpy as np
import time

# TODO Find connected clusters using the edges
# TODO find the most far points in the cluster
# TODO print ellipse instead of the cluster

# TODO fix the connected components bug
# TODO connect not connected edges


def present(image: np.ndarray):
    io.imshow(image)
    io.show()


def main():
    original = io.imread("images/4_333.tif")

    start = time.time()

    filters = ImageFilters()
    norm = filters.normalize(original)
    present(norm)

    grayed = filters.grayscale(norm)
    edges = filters.canny(grayed, gradient_ratio=10, low_threshold_ratio=0.23, high_threshold_ratio=0.27, blur_ratio=9)

    mask = filters.make_mask_background(norm, (0, 0, 0), (35, 20, 20))
    mask = filters.grayscale(mask)
    mask = filters.binary_dilation(mask,  neighborhood=10)
    edges[np.where(mask)] = 0

    edges = filters.binary_dilation(edges, neighborhood=4)
    edges = morphology.remove_small_objects(edges > 0, min_size=200)
    edges = filters.binary_dilation(edges, neighborhood=4)
    edges = filters.fill_holes(edges)

    end = time.time()
    print("process takes", (end - start)/60, "minutes")

    present(edges)


if __name__ == '__main__':
    main()
