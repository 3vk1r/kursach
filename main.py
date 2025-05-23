import numpy as np
from utils import (
    load_image_as_vector, cosine_angle, covariance_matrix,
    apply_least_squares, generate_noise_vector,
    add_noise, signal_to_noise_ratio, error_occurred
)
from report import create_pdf_report

# Загружаем 3 изображения
v1 = load_image_as_vector("img1.png")
v2 = load_image_as_vector("img2.png")
v3 = load_image_as_vector("img3.png")

# Создаём матрицы
A = np.stack([v1, v2], axis=1)
B = np.stack([v2, v3], axis=1)

# Эталонные оценки по МНК
x_ref_A1 = apply_least_squares(A, v1)
x_ref_A2 = apply_least_squares(A, v2)

# Параметры для эксперимента
iterations = 100
scale = 3.0  # коэффициент размаха шума
errors = 0
snr_values = []

# Множественные итерации
for i in range(iterations):
    noise = generate_noise_vector(size=len(v1), scale=scale)
    noisy_v1 = add_noise(v1, noise)

    # Пересчитываем результат по МНК
    x_noisy = apply_least_squares(A, noisy_v1)

    # Проверяем наличие ошибки
    if error_occurred(x_ref_A1, x_noisy):
        errors += 1

    # Считаем SNR
    snr = signal_to_noise_ratio(v1, noise)
    snr_values.append(snr)

# Вывод результатов
avg_snr = np.mean(snr_values)
error_rate = errors / iterations
img_paths = ["img1.png", "img2.png", "img3.png"]
angles = [cosine_angle(v1, v2), cosine_angle(v2, v3)]

create_pdf_report(
    output_path="report.pdf",
    img_paths=img_paths,
    angles=angles,
    snr_values=snr_values,
    error_rate=error_rate
)

print("PDF-отчёт успешно сохранён как report.pdf")
print(f"Средний SNR: {avg_snr:.2f} дБ")
print(f"Вероятность ошибки: {error_rate:.2%}")
