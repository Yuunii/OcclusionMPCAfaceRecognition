import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_shape=(32, 32)):
        self.image_shape = image_shape

    def read_images(self, path):
        images, labels = [], []
        for root, dirs, files in os.walk(path):
            for file in files:
                img_path = os.path.join(root, file)
                img = self.load_image_from_path(img_path)
                if img is not None:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_resized = cv2.resize(img_gray, self.image_shape)
                    img_normalized = img_resized / 255.0
                    images.append(img_normalized)
                    labels.append(root.split(os.path.sep)[-1])
        return images, labels

    def load_image_from_path(self, img_path):
        try:
            if img_path.lower().endswith('.gif'):
                gif = cv2.VideoCapture(img_path)
                ret, frame = gif.read()
                if ret:
                    return frame
            else:
                return cv2.imread(img_path, cv2.IMREAD_COLOR)
        except Exception as e:
            print(e)
            return None


class PCAProcessor:
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


class ModularPCAEvaluator:
    def __init__(self, images, labels, pca_processor, n):
        self.images = images
        self.labels = labels
        self.pca_processor = pca_processor
        self.n = n

    def modular_pca(self):
        correct_predictions = 0
        n_images = len(self.images)

        for leave_out_index in range(n_images):
            test_image = self.images[leave_out_index]
            test_label = self.labels[leave_out_index]
            train_images = self.images[:leave_out_index] + self.images[leave_out_index + 1:]

            # 공분산 행렬 및 고유벡터 계산
            all_patches = [self.pca_processor.split_image(img, self.n) for img in train_images]
            all_patches_flat = [patch for patches in all_patches for patch in patches]
            mean_all_patches = np.mean(all_patches_flat, axis=0)
            diff_all_patches = np.subtract(all_patches_flat, mean_all_patches)
            cov_matrix = self.pca_processor.covariance(diff_all_patches)
            mv = self.pca_processor.eigenvector(cov_matrix)

            # 각 훈련 이미지의 가중치 계산
            train_weights = []
            for img in train_images:
                patches = self.pca_processor.split_image(img, self.n)
                weights = [np.dot(np.subtract(patch, mean_all_patches), mv) for patch in patches]
                train_weights.append(weights)

            # 테스트 이미지의 가중치 계산
            test_patches = self.pca_processor.split_image(test_image, self.n)
            test_weights = [np.dot(np.subtract(patch, mean_all_patches), mv) for patch in test_patches]

            # 유사도 계산
            distances = [sum(np.linalg.norm(test_weights[j] - weights[j]) for j in range(len(test_weights))) / len(test_weights) for weights in train_weights]
            min_index = np.argmin(distances)
            predicted_label = self.labels[min_index if min_index < leave_out_index else min_index + 1]

            print(f"Test image {test_label} is most similar to training image {predicted_label}")

            if predicted_label == test_label:
                correct_predictions += 1

        recognition_rate = correct_predictions / n_images
        print(f"Recognition Rate: {recognition_rate * 100:.2f}%")
        return recognition_rate


def main():
    base_path = 'Yaledatabase_full/data'
    k = 50
    n = 16

    processor = ImageProcessor()
    images, labels = processor.read_images(base_path)

    pca_processor = PCAProcessor(k)
    evaluator = ModularPCAEvaluator(images, labels, pca_processor, n)

    recognition_rate = evaluator.modular_pca()
    print(f'MPCA16 recognition rate: {recognition_rate * 100:.2f}%')


if __name__ == '__main__':
    main()