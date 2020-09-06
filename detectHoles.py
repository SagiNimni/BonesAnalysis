from matplotlib import pyplot
from manipulations.Filters import ImageFilters
from skimage import io
import numpy as np
import time

# TODO Find connected clusters using the edges
# TODO find the most far points in the cluster
# TODO print ellipse instead of the cluster

# TODO fix the connected components bug
# TODO connect not connceted edges


def main():
    original = io.imread("images/4_333.tif")

    filters = ImageFilters(original)

    start = time.time()
    edges = filters.detect_edges(low_threshold_ratio=0.27, high_threshold_ratio=0.3)
    filters.connected_components()
    end = time.time()
    print("process takes", (end - start)/60, "minutes")

    labels = filters.remove_small_objects(low_threshold=15)

    io.imshow(labels)
    io.show()


if __name__ == '__main__':
    main()
