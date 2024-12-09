import cv2
import os
from Gaussian_Sobel_filter import ProcessingFilter

class PCA_Imageload:
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




class MPCA_GS_imageload(ProcessingFilter):
    def __init__(self, image_shape=(32, 32)):
        self.image_shape = image_shape

    def read_images(self, path):
        images, labels = [], []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(('.gif', '.jpg', '.png')):
                    img_path = os.path.join(root, file)
                    img = self.load_image_from_path(img_path)
                    if img is not None:
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img_resized = cv2.resize(img_gray, self.image_shape)
                        img_blurred = self.gaussian_blur(img_resized, kernel_size=5, sigmax=1, sigmay=2.5)
                        img_sobel = self.sobel_filters(img_blurred, axis='y')
                        img_normalized = img_sobel / 255.0
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