import numpy as np
from scipy.ndimage import convolve


class ProcessingFilter():
    def gaussian_blur(self, image, kernel_size, sigmax, sigmay):
        x, y = np.mgrid[-kernel_size//2 + 1:kernel_size//2 + 1, -kernel_size//2 + 1:kernel_size//2 + 1]
        g = np.exp(-((x**2 / (2.0 * sigmax**2)) + (y**2 / (2.0 * sigmay**2))))
        g /= g.sum()
        return convolve(image, g)

    def sobel_filters(self, image, axis='both'):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
        Ix, Iy = np.zeros_like(image, dtype=float), np.zeros_like(image, dtype=float)

        rows, cols = image.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                region = image[i - 1:i + 2, j - 1:j + 2]
                if axis in ('x', 'both'):
                    Ix[i, j] = np.sum(Kx * region)
                if axis in ('y', 'both'):
                    Iy[i, j] = np.sum(Ky * region)

        if axis == 'x':
            G = Ix
        elif axis == 'y':
            G = Iy
        else:
            G = np.hypot(Ix, Iy)

        return np.uint8(G / G.max() * 255)

