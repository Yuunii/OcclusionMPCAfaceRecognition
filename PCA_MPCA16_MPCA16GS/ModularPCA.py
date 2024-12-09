from Imageload import PCA_Imageload
from Evaluator import MPCAEvaluator
from eigenvector import MPCA


def main():
    base_path = 'Yaledatabase_full/data'
    k = 30
    n = 16

    processor = PCA_Imageload()
    images, labels = processor.read_images(base_path)

    pca_processor = MPCA(k)
    evaluator = MPCAEvaluator(images, labels, pca_processor, n)

    recognition_rate = evaluator.modular_pca()
    print(f'MPCA16 recognition rate: {recognition_rate * 100:.2f}%')


if __name__ == '__main__':
    main()