from matplotlib import pyplot
from skimage import io
from Filters import ImageFilters


def main():
    original = io.imread("images/4_333.tif")
    filters = ImageFilters(original)
    edges = filters.detect_edges()

    io.imshow(edges, cmap=pyplot.get_cmap('gray'))
    io.show()


if __name__ == '__main__':
    main()
