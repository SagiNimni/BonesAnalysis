from manipulations.Filters import ImageFilters
from skimage import io, morphology
import numpy as np
import time

# TODO Find connected clusters using the edges
# TODO find the most far points in the cluster
# TODO print ellipse instead of the cluster

# TODO fix the connected components bug
# TODO connect not connected edges


def present(image: np.ndarray, grayscale=False):
    if grayscale:
        io.imshow(image, cmap='gray')
    else:
        io.imshow(image)
    io.show()


def main():
    original = io.imread("images/4_333.tif")

    start = time.time()

    filters = ImageFilters()
    norm = filters.normalize(original)

    mask = filters.make_mask_background(norm, (0, 0, 0), (30, 20, 20))
    mask = filters.grayscale(mask)
    mask = morphology.remove_small_objects(mask > 0, min_size=150)
    mask = filters.binary_dilation(mask, neighborhood=15)

    grayed = filters.grayscale(norm)
    edges = filters.canny(grayed, mask=mask, gradient_ratio=16, low_threshold_ratio=0.23, high_threshold_ratio=0.27, blur_ratio=17)

    hard_edegs = filters.binary_dilation(edges, neighborhood=3)
    edges[np.where(hard_edegs == True)] = 255
    edges[np.where(hard_edegs == False)] = 0
    del hard_edegs

    edges = filters.remove_shapes_inside_shape(edges, shape_size=10)
    edges = morphology.remove_small_objects(edges, min_size=15)
    edges = filters.binary_dilation(edges, neighborhood=2)
    edges = morphology.remove_small_objects(edges, min_size=50)
    holes = filters.fill_holes(edges)
    original[np.where(holes)] = [0, 255, 0]

    end = time.time()
    print("process takes", (end - start)/60, "minutes")

    present(original)


if __name__ == '__main__':
    main()
