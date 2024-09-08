import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f'Не удалось загрузить изображение: {image_path}')
        return None

    # Примените Local Binary Pattern (LBP) для извлечения признаков
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    return lbp

def compare_images(features1, features2):
    # Преобразуйте изображения в гистограммы
    hist1, _ = np.histogram(features1.ravel(), bins=np.arange(0, 256))
    hist2, _ = np.histogram(features2.ravel(), bins=np.arange(0, 256))
    
    # Нормализуйте гистограммы
    hist1 = hist1.astype('float32')
    hist2 = hist2.astype('float32')
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()
    
    # Используйте корреляцию для сравнения
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

def main():
    query_image_path = 'D:/Project/python/FaceRecognition/08092024/fpDB/118.tif'
    folder_path = 'D:/Project/python/FaceRecognition/08092024/fpDB/3'

    query_features = extract_features(query_image_path)
    if query_features is None:
        print('Не удалось извлечь признаки для изображения для сравнения.')
        return

    best_match = None
    best_match_score = -1  # Для корреляции более высокая оценка лучше

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
            image_path = os.path.join(folder_path, filename)
            features = extract_features(image_path)
            
            if features is None:
                print(f'Не удалось извлечь признаки для изображения {filename}.')
                continue
            
            score = compare_images(query_features, features)
            #print(f'{filename}: Оценка сходства = {score}')

            if score > best_match_score:
                best_match_score = score
                best_match = filename

    if best_match:
        print(f'Самое похожее изображение: {best_match} с оценкой сходства {best_match_score}')
    else:
        print('Не удалось найти похожее изображение.')

if __name__ == '__main__':
    main()
