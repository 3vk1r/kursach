import numpy as np
import os
import logging
from datetime import datetime
from utils import (
    load_image_as_matrix_and_vector,
    cosine_angle,
    apply_least_squares,
    covariance_matrix,
)
from report import create_pdf_report

# Настройка логирования
logger = logging.getLogger(__name__)

def process_images_and_generate_report(img_paths, output_pdf, size, progress_callback=None):
    logger.info("Запуск анализа изображений")

    # Структура результатов
    results = {
        'input_parameters': {
            'image_count': len(img_paths),
            'image_size': size,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'matrices': [],
        'vectors': [],
        'pairwise_analysis': [],
        'cov_matrix': None,
        'statistics': {'angles': []}
    }

    try:
        # Загрузка изображений
        if progress_callback:
            progress_callback("Начало загрузки изображений...\n")

        vectors = []
        for idx, path in enumerate(img_paths):
            matrix, vector = load_image_as_matrix_and_vector(path, size)
            results['matrices'].append(matrix.tolist())
            results['vectors'].append(vector.tolist())
            vectors.append(vector)

            if progress_callback:
                progress_callback(f"Изображение {idx + 1}/{len(img_paths)} загружено\n")

        # Вычисление ковариационной матрицы
        if progress_callback:
            progress_callback("\nВычисление ковариационной матрицы...\n")

        cov_matrix = covariance_matrix(vectors)
        results['cov_matrix'] = cov_matrix.tolist()

        if progress_callback:
            progress_callback(f"Ковариационная матрица ({cov_matrix.shape[0]}x{cov_matrix.shape[1]}) вычислена\n")

        # Попарный анализ
        if progress_callback:
            progress_callback("\nНачало попарного анализа...\n")

        for i in range(len(img_paths)):
            for j in range(i + 1, len(img_paths)):
                pair_info = {
                    'pair_id': f"{i + 1}-{j + 1}",
                    'image1_idx': i + 1,
                    'image2_idx': j + 1,
                    'vector_angle': None,
                    'residual': None
                }

                try:
                    v1 = np.array(vectors[i])
                    v2 = np.array(vectors[j])

                    # Вычисление угла
                    angle = cosine_angle(v1, v2)
                    pair_info['vector_angle'] = float(angle)
                    results['statistics']['angles'].append(angle)

                    # Решение МНК
                    A = np.column_stack([v1, v2])
                    x = apply_least_squares(A, v1, cov_matrix=cov_matrix)

                    # Вычисление невязки
                    residual = np.linalg.norm(A @ x - v1)
                    pair_info['residual'] = float(residual)

                    if progress_callback:
                        msg = (f"Пара {i + 1}-{j + 1}:\n"
                               f"Угол: {angle:.2f}°\n"
                               f"Невязка: {residual:.4f}\n"
                               "────────────────────\n")
                        progress_callback(msg)

                except Exception as e:
                    logger.error(f"Ошибка анализа пары {i + 1}-{j + 1}: {str(e)}")
                    pair_info['error'] = str(e)

                results['pairwise_analysis'].append(pair_info)

        # Генерация отчёта
        if progress_callback:
            progress_callback("\nГенерация отчета...\n")

        create_pdf_report(output_pdf, img_paths, results, size)
        logger.info(f"Отчет сохранен: {output_pdf}")
        return results

    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}", exc_info=True)
        if progress_callback:
            progress_callback(f"\nОшибка: {str(e)}\n")
        raise