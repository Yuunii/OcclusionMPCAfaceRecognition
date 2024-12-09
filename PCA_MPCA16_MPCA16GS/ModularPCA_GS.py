from eigenvector import MPCA
from Imageload import MPCA_GS_imageload
from Evaluator import MPCAEvaluator


def main():
    base_path = 'Yaledatabase_full\data'
    k = 30
    n = 16

    processor = MPCA_GS_imageload()
    images, labels = processor.read_images(base_path)

    pca_processor = MPCA(k)
    evaluator = MPCAEvaluator(images, labels, pca_processor, n)

    recognition_rate = evaluator.modular_pca()
    print(f'Gaussian blur + Sobel_y + MPCA16 recognition rate: {recognition_rate * 100:.2f}%')


if __name__ == '__main__':
    main()