import numpy as np
from PIL import Image, ImageFilter
import math

def load_image_as_matrix_and_vector(path, size, threshold=0.5):
    """Загружает изображение с преобразованием в матрицу указанного размера"""
    try:
        img = Image.open(path).convert('L')
        img = img.resize(size)
        img = img.filter(ImageFilter.SHARPEN)
        data = np.array(img) / 255.0
        binary = (data < threshold).astype(int)
        return binary, binary.flatten(order='F').reshape(-1, 1)
    except Exception as e:
        raise ValueError(f"Ошибка загрузки {path}: {str(e)}")

def format_matrix(matrix):
    """Форматирует матрицу для вывода с динамическим выравниванием"""
    return '\n'.join([' '.join([str(int(x)) for x in row]) for row in matrix])

def format_vector(vector):
    """Универсальное форматирование вектора"""
    if vector is None:
        return "None"
    try:
        if isinstance(vector, np.ndarray):
            if vector.ndim == 1:
                return '\n'.join(str(int(x)) for x in vector)
            return '\n'.join(str(int(x[0])) for x in vector)
        return '\n'.join(str(int(x)) for x in vector)
    except Exception as e:
        return f"Ошибка форматирования: {str(e)}"
def cosine_angle(v1, v2):
    """Вычисляет угол между векторами в градусах"""
    dot = np.dot(v1.T, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cos_theta = dot / (norm1 * norm2)
    return np.degrees(np.arccos(cos_theta))

def covariance_matrix(vectors):
    """Вычисляет матрицу ковариации"""
    return np.cov(np.stack([v.flatten() for v in vectors]), rowvar=False)

def apply_least_squares(A, b):
    """Применяет метод наименьших квадратов"""
    return np.linalg.lstsq(A, b, rcond=None)[0]

def generate_noise_vector(size, scale=1.0):
    """Генерирует вектор шума"""
    noise = np.random.rand(size) - 0.5
    return noise * scale

def add_noise(signal, noise):
    """Добавляет шум к сигналу"""
    return signal + noise.reshape(signal.shape)

def signal_to_noise_ratio(signal, noise):
    """Вычисляет SNR в дБ"""
    signal_power = np.sum(signal ** 2)
    noise_power = np.sum(noise ** 2)
    return 10 * np.log10(signal_power / (noise_power + 1e-10))

def error_occurred(x_ref, x_noisy, threshold=0.1):
    """Проверяет наличие ошибки"""
    return np.linalg.norm(x_ref - x_noisy) > threshold