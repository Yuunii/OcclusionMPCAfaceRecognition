import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class Imageload:
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
                return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(e)
            return None

class PCAFaceRecognizer:
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


class FaceRecognitionEvaluator:
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


def main():
    base_path = 'Yaledatabase_full/data'
    k = 30

    # Initialize classes
    processor = Imageload()
    recognizer = PCAFaceRecognizer(k=k)

    # Load and preprocess images
    images, labels = processor.read_images(base_path)
    evaluator = FaceRecognitionEvaluator(recognizer, images, labels)

    # Perform leave-one-out evaluation
    recognition_rate = evaluator.leave_one_out()
    print(f'PCA recognition rate: {recognition_rate * 100:.2f}%')


if __name__ == '__main__':
    main()