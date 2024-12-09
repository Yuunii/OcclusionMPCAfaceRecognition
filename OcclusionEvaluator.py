import matplotlib.pyplot as plt
import cv2
import numpy as np
from Evaluator import MPCAEvaluator

class Occlusion():
    def mask_occlusion(self, image, top_left_y):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        top_left = (10, top_left_y)
        bottom_right = (26, 26)
        cv2.rectangle(mask, top_left, bottom_right, (255), thickness=cv2.FILLED)
        occluded_image = cv2.bitwise_and(image, image, mask=255 - mask)
        occlusion_percentage = (np.sum(mask == 255) / (image.shape[0] * image.shape[1])) * 100
        return occluded_image, occlusion_percentage


class PCAOcclusionEvaluator(Occlusion):
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



class MPCAOcclusionEvaluator(Occlusion):
    def __init__(self, image_processor, pca_evaluator, k, n):
        self.image_processor = image_processor
        self.pca_evaluator = pca_evaluator
        self.k = k
        self.n = n

    def plot_occlusion_vs_recognition(self):
        top_left_y_values = range(0, 30, 5)
        recognition_rates, occlusion_percentages = [], []

        for top_left_y in top_left_y_values:
            occluded_images, occlusion_percentage = [], 0
            for img in self.pca_evaluator.images:
                occluded_img, occlusion_percentage = self.image_processor.mask_occlusion(img, top_left_y)
                occluded_images.append(occluded_img)
            occlusion_percentages.append(occlusion_percentage)

            recognition_rate = MPCAEvaluator(occluded_images, self.pca_evaluator.labels, self.pca_evaluator.pca_processor, self.n).modular_pca()
            recognition_rates.append(recognition_rate * 100)

            print(f"Occlusion {occlusion_percentage}% - Recognition Rate: {recognition_rate * 100:.2f}%")

        plt.plot(occlusion_percentages, recognition_rates, marker='o')
        plt.xlabel('Occlusion Percentage (%)')
        plt.ylabel('Recognition Rate (%)')
        plt.grid(True)
        plt.show()

