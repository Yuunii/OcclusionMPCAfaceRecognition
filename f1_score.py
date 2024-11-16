import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage import measure

def loadImageFromPath(imgPath):
    try:
        if str(imgPath).lower().endswith('.gif'):
            gif = cv2.VideoCapture(imgPath)
            ret, frame = gif.read()
            if ret:
                return frame
        else:
            return cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(e)
        return None

def gaussian_blur(image, kernel_size=5, sigmax=1, sigmay=1):
    x, y = np.mgrid[-kernel_size//2 + 1:kernel_size//2 + 1, -kernel_size//2 + 1:kernel_size//2 + 1]
    g = np.exp(-((x**2 / (2.0 * sigmax**2)) + (y**2 / (2.0 * sigmay**2))))
    g /= g.sum()
    blurred_image = convolve(image, g)
    return blurred_image

def sobel_filters(image, axis='both'):
    """
    param
     image: 입력 이미지 입력하면 됨
     axis: 원하는 미분 축을 설정하는 파라매타

    return: 소벨필터가 적용된 이미지를 반환

    """
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)

    Ix = np.zeros_like(image, dtype=float)
    Iy = np.zeros_like(image, dtype=float)

    rows, cols = image.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            if axis == 'x' or axis == 'both':
                Ix[i, j] = np.sum(Kx * region)
            if axis == 'y' or axis == 'both':
                Iy[i, j] = np.sum(Ky * region)

    if axis == 'x':
        G = Ix
    elif axis == 'y':
        G = Iy
    else:
        G = np.hypot(Ix, Iy)

    G = G / G.max() * 255

    return np.uint8(G)


def binarize_edges(edges, threshold):
    return edges > threshold


def calculate_edge_thickness(edges):
    binary_edges = binarize_edges(edges, threshold=edges.max() / 2)
    labels = measure.label(binary_edges, connectivity=2)
    props = measure.regionprops(labels)

    thickness = ([prop.equivalent_diameter for prop in props])
    avg_thickness = np.mean(thickness)
    return avg_thickness


def main():
    path = 'Yaledatabase_full/data/1/subject01.centerlight.gif'
    img = loadImageFromPath(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gaussian + sobel
    img_gaussian = gaussian_blur(img_gray, 5,1,2)
    img_gaussian_sobel = sobel_filters(img_gaussian,axis='y')

    # only Sobel
    img_sobel = sobel_filters(img_gray,axis='y')

    # 에지 두께 계산
    original_edge_thickness = calculate_edge_thickness(img_gaussian_sobel)
    blurred_edge_thickness = calculate_edge_thickness(img_sobel)

    print(f"Gaussian + Sobel Edge Thickness: {original_edge_thickness:.2f}")
    print(f"Sobel Edge Thickness: {blurred_edge_thickness:.2f}")

if __name__ == '__main__':
    main()