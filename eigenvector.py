import numpy as np
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, k):
        self.k = k
        self.eigenfaces = None
        self.mean_train = None

    def image_as_row(self, x):
        return x.flatten()

    def covariance(self, m):
        return np.cov(m, rowvar=False)

    def eigenvector(self, cov_matrix):
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        return eigvecs[:, :self.k]

    def fit(self, train_images):
        vector = np.array([self.image_as_row(img) for img in train_images])
        self.mean_train = vector.mean(axis=0)
        diff = vector - self.mean_train
        cov = self.covariance(diff)
        self.eigenfaces = self.eigenvector(cov)
        return np.dot(diff, self.eigenfaces)

    def transform(self, image):
        image_row = self.image_as_row(image)
        image_mean = image_row - self.mean_train
        return np.dot(image_mean, self.eigenfaces)

    def plot_eigenfaces(self, num_eigenfaces=3):
        plt.figure(figsize=(10, 4))
        for i in range(num_eigenfaces):
            eigenface = self.eigenfaces[:, i].reshape((32, 32))
            plt.subplot(1, num_eigenfaces, i + 1)
            plt.imshow(eigenface, cmap='gray')
            plt.axis('off')
        plt.show()


class MPCA:
    def __init__(self, k):
        self.k = k

    def split_image(self, image, n):
        h, w = image.shape
        patches = []
        patch_size = int(h / np.sqrt(n))
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = image[i:i + patch_size, j:j + patch_size].flatten()
                patches.append(patch)
        return patches

    def covariance(self, matrix):
        return np.cov(matrix, rowvar=False)

    def eigenvector(self, cov_matrix):
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigvals)[::-1]
        return eigvecs[:, idx][:, :self.k]
