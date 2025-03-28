from Imageload import Occlusion_PCA_Imageload
from eigenvector import PCA
from Evaluator import Occlusion_PCA_Evaluator
from OcclusionEvaluator import PCAOcclusionEvaluator


def main():
    base_path = 'Yaledatabase_full/data'
    k = 30
    image_shape = (32, 32)

    processor = Occlusion_PCA_Imageload(image_shape)
    images, labels = processor.read_images(base_path)

    pca_processor = PCA(k)
    face_recognizer = Occlusion_PCA_Evaluator(pca_processor)

    #오클루전 평가
    evaluator = PCAOcclusionEvaluator(face_recognizer, processor)
    evaluator.plot_occlusion_vs_recognition(images, labels, k)


if __name__ == '__main__':
    main()