import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PCA import Imageload

class Occlusion():
    def mask_occlusion(self, image, top_left_y):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        top_left = (10, top_left_y)
        bottom_right = (26, 26)
        cv2.rectangle(mask, top_left, bottom_right, (255), thickness=cv2.FILLED)
        occluded_image = cv2.bitwise_and(image, image, mask=255 - mask)
        occlusion_percentage = (np.sum(mask == 255) / (image.shape[0] * image.shape[1])) * 100
        return occluded_image, occlusion_percentage


class PCAProcessor():
    def __init__(self, k):
        self.k = k

    def image_as_row(self, image):
        return image.flatten()

    def covariance(self, matrix):
        return np.cov(matrix, rowvar=False)

    def eigenvector(self, cov_matrix):
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigvals)[::-1]
        return eigvecs[:, idx][:, :self.k]

    def plot_eigenfaces(self, eigenvectors, image_shape):
        fig, axs = plt.subplots(1, self.k, figsize=(15, 15))
        for i in range(self.k):
            eigenface = eigenvectors[:, i].reshape(image_shape)
            axs[i].imshow(eigenface, cmap='gray')
            axs[i].axis('off')
        plt.show()


class FaceRecognizer():
    def __init__(self, pca_processor):
        self.pca_processor = pca_processor

    def leave_one_out(self, images, labels):
        if not images:
            print("Error: No images to process in leave_one_out")
            return 0
        correct_predictions = 0
        n = len(images)

        for leave_out_index in range(n):
            test_image = images[leave_out_index]
            test_label = labels[leave_out_index]
            train_images = images[:leave_out_index] + images[leave_out_index + 1:]
            train_labels = labels[:leave_out_index] + labels[leave_out_index + 1:]

            vector = np.array([self.pca_processor.image_as_row(img) for img in train_images])
            mean_train = vector.mean(axis=0)
            diff = np.subtract(vector, mean_train)
            cov = self.pca_processor.covariance(diff)
            mv = self.pca_processor.eigenvector(cov)
            mapped_train = np.dot(diff, mv)

            new_image_row = self.pca_processor.image_as_row(test_image)
            new_image_mean = np.subtract(new_image_row, mean_train)
            mapped_test = np.dot(new_image_mean, mv)

            distances = [np.linalg.norm(mapped_test - train_vec) for train_vec in mapped_train]
            min_index = np.argmin(distances)
            predicted_label = train_labels[min_index if min_index < leave_out_index else min_index + 1]
            if predicted_label == test_label:
                correct_predictions += 1

        return correct_predictions / n


class OcclusionEvaluator(Occlusion):
    def __init__(self, face_recognizer, image_processor):
        self.face_recognizer = face_recognizer
        self.image_processor = image_processor

    def plot_occlusion_vs_recognition(self, images, labels, k):
        top_left_y_values = range(0, 30, 5)
        recognition_rates, occlusion_percentages = [], []

        for top_left_y in top_left_y_values:
            occluded_images, occlusion_percentage = [], 0
            for img in images:
                occluded_img, occlusion_percentage = self.image_processor.mask_occlusion(img, top_left_y)
                occluded_images.append(occluded_img)
            occlusion_percentages.append(occlusion_percentage)

            recognition_rate = self.face_recognizer.leave_one_out(occluded_images, labels)
            recognition_rates.append(recognition_rate * 100)
            print(f"Occlusion {occlusion_percentage}% - Recognition Rate: {recognition_rate * 100:.2f}%")

        plt.plot(occlusion_percentages, recognition_rates, marker='o')
        plt.xlabel('Occlusion Percentage (%)')
        plt.ylabel('Recognition Rate (%)')
        plt.grid(True)
        plt.show()


def main():
    base_path = 'Yaledatabase_full/data'
    k = 30
    image_shape = (32, 32)

    processor = Imageload(image_shape)
    images, labels = processor.read_images(base_path)

    pca_processor = PCAProcessor(k)
    face_recognizer = FaceRecognizer(pca_processor)

    #오클루전 평가
    evaluator = OcclusionEvaluator(face_recognizer, processor)
    evaluator.plot_occlusion_vs_recognition(images, labels, k)

"""
    # 고유얼굴 시각화
    vector = np.array([pca_processor.image_as_row(img) for img in images])
    mean_train = vector.mean(axis=0)
    diff = np.subtract(vector, mean_train)
    cov = pca_processor.covariance(diff)
    mv = pca_processor.eigenvector(cov)
    pca_processor.plot_eigenfaces(mv, image_shape)


"""


if __name__ == '__main__':
    main()