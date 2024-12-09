import numpy as np


class PCAEvaluator:
    def __init__(self, recognizer, images, labels):
        self.recognizer = recognizer
        self.images = images
        self.labels = labels

    def find_similar(self, image, mapped_train):
        distances = [np.linalg.norm(image - m) for m in mapped_train]
        return np.argmin(distances)

    def leave_one_out(self):
        correct_predictions = 0
        n = len(self.images)

        for leave_out_index in range(n):
            test_image = self.images[leave_out_index]
            test_label = self.labels[leave_out_index]
            train_images = self.images[:leave_out_index] + self.images[leave_out_index + 1:]
            train_labels = self.labels[:leave_out_index] + self.labels[leave_out_index + 1:]

            mapped_train = self.recognizer.fit(train_images)

            # Plot eigenfaces once
            if leave_out_index == 0:
                self.recognizer.plot_eigenfaces()

            mapped_test = self.recognizer.transform(test_image)
            index = self.find_similar(mapped_test, mapped_train)
            predicted_label = train_labels[index if index < leave_out_index else index + 1]
            print(f"Test image {test_label} is most similar to training image {index}")

            if predicted_label == test_label:
                correct_predictions += 1

        recognition_rate = correct_predictions / n
        print(f"Recognition Rate: {recognition_rate * 100:.2f}%")
        return recognition_rate



class MPCAEvaluator:
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
