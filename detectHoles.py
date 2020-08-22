from matplotlib import pyplot
from Filters import ImageFilters
from skimage import io

# TODO Find connected clusters using the edges
# TODO find the most far points in the cluster
# TODO print ellipse instead of the cluster


def main():
    original = io.imread("images/4_333.tif")

    filters = ImageFilters(original)
    edges = filters.detect_edges(low_threshold_ratio=0.28, high_threshold_ratio=0.3)

    io.imshow(edges, cmap=pyplot.get_cmap('gray'))
    io.show()


if __name__ == '__main__':
    main()
