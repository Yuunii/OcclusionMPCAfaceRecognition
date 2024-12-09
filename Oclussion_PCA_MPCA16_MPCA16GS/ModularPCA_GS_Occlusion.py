from Imageload import MPCA_GS_imageload
from eigenvector import MPCA
from Evaluator import MPCAEvaluator
from OcclusionEvaluator import MPCAOcclusionEvaluator


def main():
    base_path = 'Yaledatabase_full/data'
    k = 30
    n = 16

    processor = MPCA_GS_imageload()
    images, labels = processor.read_images(base_path)

    pca_processor = MPCA(k)
    pca_evaluator = MPCAEvaluator(images, labels, pca_processor, n)

    # 오클루전 평가
    evaluator = MPCAOcclusionEvaluator(processor, pca_evaluator, k, n)
    evaluator.plot_occlusion_vs_recognition()



if __name__ == '__main__':
    main()