import numpy as np
from PIL import Image, ImageFilter
import logging
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler('utils.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_image_as_matrix_and_vector(path, size, threshold=0.5):
    """Загрузка и обработка изображения"""
    logger.info(f"Загрузка изображения: {path}")
    try:
        img = Image.open(path).convert('L')
        img = img.resize(size)
        img = img.filter(ImageFilter.SHARPEN)

        data = np.array(img) / 255.0
        binary = (data < threshold).astype(int)
        vector = binary.flatten(order='F')

        logger.info(f"Изображение загружено. Размер: {binary.shape}")
        return binary, vector

    except Exception as e:
        logger.critical(f"Ошибка загрузки {path}: {e}")
        raise

def cosine_angle(v1, v2):
    """Вычисление угла между векторами"""
    logger.debug("Вычисление угла между векторами")
    try:
        v1 = v1.flatten().astype(float)
        v2 = v2.flatten().astype(float)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 90.0

        cos_theta = dot_product / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))

        logger.info(f"Угол: {angle:.2f}°")
        return angle

    except Exception as e:
        logger.error(f"Ошибка вычисления угла: {str(e)}")
        raise

def covariance_matrix(vectors):
    """Вычисление ковариационной матрицы"""
    logger.info("Вычисление ковариационной матрицы")
    try:
        stacked = np.stack([v.flatten() for v in vectors])
        cov = np.cov(stacked, rowvar=False)
        logger.info(f"Ковариационная матрица: {cov.shape}")
        return cov
    except Exception as e:
        logger.error(f"Ошибка вычисления ковариации: {e}")
        raise

def apply_least_squares(A, b, cov_matrix=None):
    """Обобщённый метод наименьших квадратов"""
    logger.debug("Применение МНК")
    try:
        A = A.astype(float)
        b = b.astype(float).flatten()

        if cov_matrix is None:
            # Стандартный МНК
            x = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            # Обобщённый МНК с ковариацией
            cov_matrix = cov_matrix.astype(float)
            try:
                cov_inv = np.linalg.inv(cov_matrix)
            except:
                cov_inv = np.linalg.pinv(cov_matrix)

            left = A.T @ cov_inv @ A
            right = A.T @ cov_inv @ b
            x = np.linalg.solve(left, right)

        logger.info("МНК завершён")
        return x

    except Exception as e:
        logger.error(f"Ошибка МНК: {str(e)}")
        raise