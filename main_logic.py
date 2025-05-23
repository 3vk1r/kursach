# main_logic.py

from utils import (
    load_image_as_vector, cosine_angle, covariance_matrix,
    apply_least_squares, generate_noise_vector,
    add_noise, signal_to_noise_ratio, error_occurred
)
from report import create_pdf_report
import numpy as np


def process_images_and_generate_report(img_paths, scale, output_pdf, size):
    print(f"\nЗагрузка {len(img_paths)} изображений...")
    vectors = []
    for i, path in enumerate(img_paths):
        try:
            vec = load_image_as_vector(path, size=size)
            vectors.append(vec)
            print(f"{i + 1}. {path} - успешно (размер: {len(vec)} пикселей)")
        except Exception as e:
            print(f"{i + 1}. {path} - ошибка: {str(e)}")
            continue

    if len(vectors) < 2:
        raise ValueError("Требуется минимум 2 корректных изображения")

    print("\nРасчёт углов между символами:")
    angles = []
    for i in range(len(vectors) - 1):
        angle = cosine_angle(vectors[i], vectors[i + 1])
        angles.append(angle)
        print(f"Угол между символами {i + 1} и {i + 2}: {angle:.2f}°")

    print("\nАнализ ошибок распознавания:")
    error_count = 0
    total_iterations = 100
    snr_values = []

    for i in range(len(vectors) - 1):
        A = np.stack([vectors[i], vectors[i + 1]], axis=1)
        x_ref = apply_least_squares(A, vectors[i])

        for j in range(total_iterations):
            noise = generate_noise_vector(len(vectors[i]), scale)
            noisy_vector = add_noise(vectors[i], noise)
            x_noisy = apply_least_squares(A, noisy_vector)

            if error_occurred(x_ref, x_noisy):
                error_count += 1

            snr = signal_to_noise_ratio(vectors[i], noise)
            snr_values.append(snr)

    error_rate = error_count / (total_iterations * (len(vectors) - 1))
    avg_snr = np.mean(snr_values)

    print("\nИтоговые результаты:")
    print(f"Средний SNR: {avg_snr:.2f} дБ")
    print(f"Общая вероятность ошибки: {error_rate:.2%}")
    print(f"\nГенерация отчёта: {output_pdf}")

    create_pdf_report(output_pdf, img_paths, angles, snr_values, error_rate, size)
