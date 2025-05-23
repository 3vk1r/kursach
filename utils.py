# utils.py

import numpy as np
from PIL import Image
import math

def load_image_as_vector(path, size=(7, 5), strict_check=False):
    img = Image.open(path).convert('L')
    if strict_check and img.size != (size[0], size[1]):
        raise ValueError(f"Размер изображения {path} — {img.size}, должен быть {size}")

    img = img.resize(size)
    data = np.asarray(img) / 255.0
    data = 1.0 - data
    return data.flatten(order='F')

def cosine_angle(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cos_theta = dot / (norm1 * norm2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle_rad)

def covariance_matrix(vectors):
    return np.cov(np.stack(vectors), rowvar=False)

def apply_least_squares(A, b):
    return np.linalg.lstsq(A, b, rcond=None)[0]

def generate_noise_vector(size, scale=1.0):
    noise = np.random.rand(size) - 0.5  # значения от -0.5 до 0.5
    return noise * scale

def add_noise(vector, noise):
    return vector + noise

def signal_to_noise_ratio(signal, noise):
    signal_power = np.sum(signal ** 2)
    noise_power = np.sum(noise ** 2)
    return 10 * np.log10(signal_power / (noise_power + 1e-10))  # dB

def error_occurred(original_result, noisy_result, threshold=0.1):
    return np.linalg.norm(original_result - noisy_result) > threshold