from Imageload import PCA_Imageload
from Evaluator import PCAEvaluator
from eigenvector import PCA


def main():
    base_path = 'Yaledatabase_full/data'
    k = 30

    # Initialize classes
    processor = PCA_Imageload()
    recognizer = PCA(k)

    # Load and preprocess images
    images, labels = processor.read_images(base_path)
    evaluator = PCAEvaluator(recognizer, images, labels)

    # Perform leave-one-out evaluation
    recognition_rate = evaluator.leave_one_out()
    print(f'PCA recognition rate: {recognition_rate * 100:.2f}%')


if __name__ == '__main__':
    main()