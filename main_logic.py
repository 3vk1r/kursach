import numpy as np
from utils import *
from report import create_pdf_report
import os


def process_images_and_generate_report(img_paths, output_pdf, size, progress_callback=None):
    """Основная функция анализа с динамическими размерами"""
    # Проверка допустимости размеров
    width, height = size
    if width < 3 or height < 3 or width > 100 or height > 100:
        raise ValueError("Допустимые размеры: от 3x3 до 100x100")

    results = {
        'matrices': [],
        'vectors': [],
        'angles': {},
        'cov_matrices': [],
        'snr_vs_error': [],
        'error_rates': [],
        'used_size': size
    }

    # Загрузка изображений
    for i, path in enumerate(img_paths):
        matrix, vector = load_image_as_matrix_and_vector(path, size)
        results['matrices'].append(matrix)
        results['vectors'].append(vector)

        if progress_callback:
            progress_callback(f"Изображение {i + 1} ({size[0]}x{size[1]}):\n"
                              f"Матрица:\n{format_matrix(matrix)}\n"
                              f"Вектор:\n{format_vector(vector)}\n")

    # Анализ пар изображений
    for i in range(len(img_paths)):
        for j in range(i + 1, len(img_paths)):
            v1 = results['vectors'][i]
            v2 = results['vectors'][j]

            # Проверка совместимости размеров
            if v1.shape != v2.shape:
                raise ValueError(f"Несовпадение размеров векторов {i + 1} и {j + 1}")

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

            if progress_callback:
                progress_callback(
                    f"Сравнение {i + 1}-{j + 1}:\n"
                    f"Угол: {angle:.2f}°\n"
                    f"Ковариация:\n{format_matrix(cov)}\n"
                    f"Ошибки: {error_rate:.1%}\n"
                    f"SNR: {avg_snr:.1f} дБ\n"
                    "----------------------------\n"
                )

    # Создание PDF отчета
    try:
        create_pdf_report(output_pdf, img_paths, results, size)
        if not os.path.exists(output_pdf):
            raise RuntimeError("Файл отчета не был создан")
    except Exception as e:
        if progress_callback:
            progress_callback(f"Ошибка создания отчета: {str(e)}")
        raise

    return results