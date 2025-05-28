import numpy as np
import os
import logging
from datetime import datetime
from utils import (
    load_image_as_matrix_and_vector,
    cosine_angle,
    apply_least_squares,
    generate_noise_vector,
    add_noise,
    signal_to_noise_ratio,
    error_occurred
)

# Настройка логирования
logger = logging.getLogger(__name__)


def process_images_and_generate_report(img_paths, output_pdf, size, noise_scale=3.0, progress_callback=None):
    """
    Основная функция анализа изображений с генерацией отчета

    Параметры:
    img_paths (list): Список путей к изображениям
    output_pdf (str): Путь для сохранения PDF-отчета
    size (tuple): Размер матрицы (ширина, высота)
    noise_scale (float): Уровень шума (0.1-5.0)
    progress_callback (function): Функция для передачи сообщений о прогрессе

    Возвращает:
    dict: Результаты анализа
    """

    logger.info("Запуск анализа изображений")

    # Инициализация структуры результатов
    results = {
        'input_parameters': {
            'image_count': len(img_paths),
            'image_size': size,
            'noise_level': noise_scale,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'matrices': [],
        'vectors': [],
        'pairwise_analysis': [],
        'statistics': {
            'angles': [],
            'snr_values': [],
            'error_rates': [],
            'snr_error_data': []  # Для хранения данных для графика
        }
    }

    try:
        # Этап 1: Загрузка и подготовка изображений
        if progress_callback:
            progress_callback("Начало загрузки изображений...\n")

        expected_length = size[0] * size[1]

        for idx, path in enumerate(img_paths):
            try:
                # Загрузка изображения
                matrix, vector = load_image_as_matrix_and_vector(path, size)

                # Проверка размера вектора
                if len(vector) != expected_length:
                    raise ValueError(f"Некорректный размер вектора: {len(vector)} вместо {expected_length}")

                # Сохранение результатов
                results['matrices'].append(matrix.tolist())
                results['vectors'].append(vector)

                if progress_callback:
                    progress_callback(f"Изображение {idx + 1}/{len(img_paths)} загружено\n")

            except Exception as e:
                logger.error(f"Ошибка загрузки {path}: {str(e)}")
                raise

        # Этап 2: Попарный анализ изображений
        if progress_callback:
            progress_callback("\nНачало попарного анализа...\n")

        for i in range(len(img_paths)):
            for j in range(i + 1, len(img_paths)):
                pair_id = f"{i + 1}-{j + 1}"
                pair_info = None

                try:
                    # Инициализация структуры для пары
                    pair_info = {
                        'pair_id': pair_id,
                        'image1': {
                            'path': img_paths[i],
                            'index': i + 1,
                            'name': os.path.basename(img_paths[i])
                        },
                        'image2': {
                            'path': img_paths[j],
                            'index': j + 1,
                            'name': os.path.basename(img_paths[j])
                        },
                        'vector_angle': None,
                        'error_rate': None,
                        'avg_snr': None
                    }

                    # Получение векторов
                    v1 = np.array(results['vectors'][i]).flatten()
                    v2 = np.array(results['vectors'][j]).flatten()

                    # Проверка размерности
                    if v1.shape != v2.shape:
                        raise ValueError(f"Несовпадение размеров векторов: {v1.shape} vs {v2.shape}")

                    # Вычисление угла между векторами
                    angle = cosine_angle(v1, v2)
                    pair_info['vector_angle'] = float(angle)
                    results['statistics']['angles'].append(angle)

                    # Создание матрицы для МНК
                    A = np.column_stack([v1, v2])
                    b = v1.copy()

                    # Проверка размеров матрицы
                    if A.shape[0] != b.shape[0]:
                        raise ValueError(f"Несовместимые размеры: A {A.shape}, b {b.shape}")

                    # Эталонное решение
                    x_ref = apply_least_squares(A, b)

                    # Проведение экспериментов с шумом
                    errors = 0
                    snr_values = []
                    n_experiments = 100  # Увеличили количество итераций

                    for exp_num in range(n_experiments):
                        try:
                            # Генерация шума
                            current_noise = noise_scale * (0.8 + 0.4 * np.random.rand())
                            noise = generate_noise_vector(v1.size, scale=current_noise)

                            # Добавление шума
                            noisy_vector = add_noise(v1, noise)

                            # Решение с шумом
                            x_noisy = apply_least_squares(A, noisy_vector)

                            # Расчет метрик
                            error = error_occurred(x_ref, x_noisy)
                            snr = signal_to_noise_ratio(v1, noise)

                            # Подсчет ошибок
                            if error:
                                errors += 1
                            snr_values.append(snr)

                            # Сохранение для графика
                            results['statistics']['snr_error_data'].append((snr, error))

                        except Exception as e:
                            logger.error(f"Ошибка эксперимента {exp_num + 1} в паре {pair_id}: {str(e)}")
                            continue

                    # Расчет вероятности ошибки
                    pair_info['error_rate'] = errors / n_experiments
                    pair_info['avg_snr'] = np.mean(snr_values) if snr_values else 0

                    # Сохранение результатов эксперимента
                    results['statistics']['error_rates'].append(pair_info['error_rate'])
                    results['statistics']['snr_values'].extend(snr_values)

                    # Сохранение результатов анализа пары
                    results['pairwise_analysis'].append(pair_info)

                    if progress_callback:
                        msg = (f"Пара {pair_id}:\n"
                               f"Угол: {angle:.2f}°\n"
                               f"Вероятность ошибки: {pair_info['error_rate']:.2%}\n"
                               "────────────────────\n")
                        progress_callback(msg)

                except Exception as e:
                    logger.error(f"Ошибка анализа пары {pair_id}: {str(e)}")
                    if pair_info:
                        pair_info['error'] = str(e)
                        results['pairwise_analysis'].append(pair_info)
                    continue

        # Этап 3: Генерация отчета
        if progress_callback:
            progress_callback("\nГенерация отчета...\n")

        from report import create_pdf_report
        create_pdf_report(output_pdf, img_paths, results, size)

        logger.info(f"Отчет успешно сохранен: {output_pdf}")
        return results

    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}", exc_info=True)
        if progress_callback:
            progress_callback(f"\nОшибка: {str(e)}\n")
        raise