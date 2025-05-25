import numpy as np
from utils import *
from report import create_pdf_report
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_images_and_generate_report(img_paths, output_pdf, size, progress_callback=None):
    """Основная функция с защитой от ошибок форматирования"""
    try:
        logger.info(f"Начало обработки. Изображений: {len(img_paths)}, размер: {size}")

        # Проверка входных данных
        if len(img_paths) < 2:
            error_msg = f"Требуется 2+ изображений, получено {len(img_paths)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        results = {
            'matrices': [],
            'vectors': [],
            'angles': {},
            'cov_matrices': [],
            'snr_vs_error': [],
            'error_rates': []
        }

        # Обработка изображений
        for i, path in enumerate(img_paths):
            try:
                logger.debug(f"Обработка изображения {i + 1}: {path}")
                matrix, vector = load_image_as_matrix_and_vector(path, size)

                # Явное преобразование в список
                matrix = matrix.astype(int).tolist()
                vector = vector.astype(int).tolist()

                results['matrices'].append(matrix)
                results['vectors'].append(vector)
                logger.debug(f"Изображение {i + 1} обработано успешно")

                if progress_callback:
                    progress_callback(
                        f"Изображение {i + 1}:\n"
                        f"Матрица:\n{format_matrix(matrix)}\n"
                        f"Вектор:\n{format_vector(vector)}\n"
                    )

            except Exception as e:
                logger.error(f"Ошибка обработки изображения {path}: {e}")
                raise

        # Анализ пар изображений
        for i in range(len(img_paths)):
            for j in range(i + 1, len(img_paths)):
                try:
                    logger.debug(f"Сравнение изображений {i + 1} и {j + 1}")
                    v1 = np.array(results['vectors'][i])
                    v2 = np.array(results['vectors'][j])

                    # Проверка и преобразование размерностей
                    if v1.ndim == 1:
                        v1 = v1.reshape(-1, 1)
                    if v2.ndim == 1:
                        v2 = v2.reshape(-1, 1)

                    # Проверка совместимости размеров
                    if v1.shape != v2.shape:
                        error_msg = f"Несовпадение размеров векторов {i + 1} и {j + 1}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    # Создание матрицы A
                    A = np.hstack([v1, v2])
                    logger.debug(f"Матрица A создана. Shape: {A.shape}")
                    angle = cosine_angle(v1, v2)
                    results['angles'][f"{i + 1}-{j + 1}"] = angle

                    cov = covariance_matrix([v1, v2])
                    results['cov_matrices'].append(cov)

                    # Эксперимент с шумом
                    A = np.hstack([v1, v2])
                    x_ref = apply_least_squares(A, v1)

                    errors = 0
                    snr_values = []
                    for _ in range(100):
                        noise = generate_noise_vector(len(v1), scale=3.0)
                        noisy_v = add_noise(v1, noise)
                        x_noisy = apply_least_squares(A, noisy_v)

                        if error_occurred(x_ref, x_noisy):
                            errors += 1
                        snr_values.append(signal_to_noise_ratio(v1, noise))

                    error_rate = errors / 100
                    avg_snr = np.mean(snr_values)
                    results['snr_vs_error'].append((avg_snr, error_rate))
                    results['error_rates'].append(error_rate)

                    logger.debug(f"Сравнение {i + 1}-{j + 1} завершено. Угол: {angle:.2f}°, Ошибки: {error_rate:.1%}")

                    if progress_callback:
                        progress_callback(
                            f"Сравнение {i + 1}-{j + 1}:\n"
                            f"Угол: {angle:.2f}°\n"
                            f"Ковариация:\n{format_matrix(cov)}\n"
                            f"Ошибки: {error_rate:.1%}\n"
                            f"SNR: {avg_snr:.1f} дБ\n"
                            "----------------------------\n"
                        )

                except Exception as e:
                    logger.error(f"Ошибка сравнения изображений {i + 1} и {j + 1}: {e}")
                    raise

        # Создание PDF отчета
        try:
            logger.info("Создание PDF отчета")
            create_pdf_report(output_pdf, img_paths, results, size)
            if not os.path.exists(output_pdf):
                error_msg = "Файл отчета не был создан"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            logger.info("Отчет успешно создан")
        except Exception as e:
            logger.error(f"Ошибка создания отчета: {e}")
            if progress_callback:
                progress_callback(f"Ошибка создания отчета: {str(e)}")
            raise

        return results

    except Exception as e:
        logger.critical(f"Критическая ошибка в процессе анализа: {e}")
        raise