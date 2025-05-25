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


def safe_format(value):
    """Безопасное преобразование значения в строку с логированием"""
    try:
        logger.debug(f"Преобразование значения: {value} (тип: {type(value)})")

        if isinstance(value, np.integer):
            result = str(int(value))
        elif isinstance(value, np.floating):
            result = str(float(value))
        else:
            result = str(int(float(value)))

        logger.debug(f"Успешно преобразовано в: {result}")
        return result

    except (ValueError, TypeError) as e:
        logger.error(f"Ошибка преобразования значения {value}: {e}")
        return str(value)


def format_matrix(matrix):
    """Форматирование матрицы в строку с полным логированием"""
    logger.info("Начало форматирования матрицы")
    try:
        logger.debug(
            f"Входные данные: {matrix} (тип: {type(matrix)}, shape: {np.shape(matrix) if hasattr(matrix, 'shape') else 'N/A'})")

        # Преобразование numpy array в список
        if isinstance(matrix, np.ndarray):
            logger.debug("Обнаружен numpy array - преобразование")
            matrix = matrix.astype(int).tolist()
            logger.debug(
                f"После преобразования: {matrix} (тип элементов: {type(matrix[0][0]) if len(matrix) > 0 else 'empty'})")

        # Проверка структуры данных
        if not all(isinstance(row, (list, tuple)) for row in matrix):
            logger.error("Некорректная структура матрицы")
            raise ValueError("Матрица должна быть двумерным массивом")

        # Форматирование каждой строки
        formatted_rows = []
        for i, row in enumerate(matrix):
            try:
                formatted_row = ' '.join([safe_format(x) for x in row])
                formatted_rows.append(formatted_row)
                logger.debug(f"Строка {i} отформатирована: {formatted_row[:50]}...")
            except Exception as e:
                logger.error(f"Ошибка форматирования строки {i}: {row}")
                raise

        result = '\n'.join(formatted_rows)
        logger.info("Матрица успешно отформатирована")
        return result

    except Exception as e:
        logger.critical(f"Критическая ошибка форматирования матрицы: {e}")
        raise


def format_vector(vector):
    """Форматирование вектора в строку с детальным логированием"""
    logger.info("Начало форматирования вектора")
    try:
        logger.debug(
            f"Входные данные: {vector} (тип: {type(vector)}, len: {len(vector) if hasattr(vector, '__len__') else 'N/A'})")

        # Обработка numpy array
        if isinstance(vector, np.ndarray):
            logger.debug("Обнаружен numpy array - преобразование")
            vector = vector.flatten().tolist()

        # Преобразование вектора-столбца в одномерный список
        if isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
            logger.debug("Обнаружен вектор-столбец - преобразование в одномерный список")
            vector = [x[0] for x in vector]

        logger.debug(
            f"После преобразования: {vector} (тип элементов: {type(vector[0]) if len(vector) > 0 else 'empty'})")

        # Форматирование элементов
        formatted_items = []
        for i, item in enumerate(vector):
            try:
                formatted_item = safe_format(item)
                formatted_items.append(formatted_item)
                logger.debug(f"Элемент {i} отформатирован: {formatted_item}")
            except Exception as e:
                logger.error(f"Ошибка форматирования элемента {i}: {item}")
                raise

        result = '\n'.join(formatted_items)
        logger.info("Вектор успешно отформатирован")
        return result

    except Exception as e:
        logger.critical(f"Критическая ошибка форматирования вектора: {e}")
        raise

    except Exception as e:
        logger.critical(f"Критическая ошибка форматирования вектора: {e}")
        raise


def load_image_as_matrix_and_vector(path, size, threshold=0.5):
    """Загрузка изображения с полным трекингом операций"""
    logger.info(f"Начало загрузки изображения: {path}")
    try:
        # Открытие изображения
        logger.debug("Открытие файла изображения")
        img = Image.open(path)
        logger.debug(f"Исходный режим изображения: {img.mode}")

        # Конвертация в grayscale
        img = img.convert('L')
        logger.debug("Изображение преобразовано в grayscale")

        # Изменение размера
        logger.debug(f"Изменение размера до: {size}")
        img = img.resize(size)

        # Улучшение резкости
        img = img.filter(ImageFilter.SHARPEN)
        logger.debug("Применён фильтр резкости")

        # Преобразование в numpy array
        data = np.array(img)
        logger.debug(f"Создан numpy array, shape: {data.shape}, dtype: {data.dtype}")

        # Нормализация и бинаризация
        data = data / 255.0
        binary = (data < threshold).astype(int)
        logger.debug(f"Бинаризация завершена. Уникальные значения: {np.unique(binary)}")

        # Создание вектора
        vector = binary.flatten(order='F')
        logger.debug(f"Вектор создан. Размер: {vector.shape}")

        logger.info(f"Изображение успешно загружено. Размер матрицы: {binary.shape}")
        return binary, vector

    except Exception as e:
        logger.critical(f"Ошибка загрузки изображения {path}: {e}")
        raise ValueError(f"Ошибка загрузки {path}: {str(e)}")


def cosine_angle(v1, v2):
    """Вычисление угла между векторами с защитой от ошибок форматирования"""
    logger.debug("Вычисление косинуса угла между векторами")
    try:
        # Преобразуем в numpy arrays на случай, если это списки
        v1 = np.array(v1, dtype=np.float64)
        v2 = np.array(v2, dtype=np.float64)

        # Проверка и корректировка размерностей
        if v1.ndim == 1:
            v1 = v1.reshape(-1, 1)
        if v2.ndim == 1:
            v2 = v2.reshape(-1, 1)

        logger.debug(f"Вектор 1: shape {v1.shape}, norm {float(np.linalg.norm(v1))}")
        logger.debug(f"Вектор 2: shape {v2.shape}, norm {float(np.linalg.norm(v2))}")

        # Вычисляем скалярное произведение
        dot_product = float(np.dot(v1.T, v2))
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))

        logger.debug(f"Точечное произведение: {dot_product}, нормы: {norm1}, {norm2}")

        # Вычисляем косинус угла с защитой от деления на ноль
        denominator = norm1 * norm2
        if denominator == 0:
            logger.warning("Нулевая норма вектора - угол не определен")
            return 90.0  # Возвращаем 90 градусов как значение по умолчанию

        cos_theta = dot_product / denominator
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Защита от ошибок округления

        angle = float(np.degrees(np.arccos(cos_theta)))
        logger.info(f"Угол между векторами: {angle:.2f}°")

        return angle

    except Exception as e:
        logger.error(f"Ошибка вычисления угла: {str(e)}")
        raise ValueError(f"Ошибка вычисления угла между векторами: {str(e)}")


def covariance_matrix(vectors):
    """Вычисление матрицы ковариации с логированием"""
    logger.info("Вычисление матрицы ковариации")
    try:
        logger.debug(f"Количество векторов: {len(vectors)}, shape каждого: {vectors[0].shape}")

        stacked = np.stack([v.flatten() for v in vectors])
        logger.debug(f"Объединённая матрица: shape {stacked.shape}")

        cov = np.cov(stacked, rowvar=False)
        logger.info(f"Матрица ковариации вычислена. Shape: {cov.shape}")

        return cov

    except Exception as e:
        logger.error(f"Ошибка вычисления ковариации: {e}")
        raise


def apply_least_squares(A, b):
    """Метод наименьших квадратов с логированием и проверкой размерностей"""
    logger.info("Применение метода наименьших квадратов")
    try:
        # Преобразование в numpy array, если это еще не сделано
        A = np.array(A) if not isinstance(A, np.ndarray) else A
        b = np.array(b) if not isinstance(b, np.ndarray) else b

        # Проверка и корректировка размерностей
        if A.ndim == 1:
            logger.debug("A одномерный - преобразование в столбец")
            A = A.reshape(-1, 1)
        if b.ndim == 1:
            logger.debug("b одномерный - преобразование в столбец")
            b = b.reshape(-1, 1)

        logger.debug(f"Матрица A: shape {A.shape}")
        logger.debug(f"Вектор b: shape {b.shape}")

        # Проверка совместимости размеров
        if A.shape[0] != b.shape[0]:
            logger.error(f"Несовместимые размеры: A {A.shape}, b {b.shape}")
            raise ValueError("Размеры A и b по оси 0 должны совпадать")

        result = np.linalg.lstsq(A, b, rcond=None)[0]
        logger.info(f"MHK решение получено. Shape: {result.shape}")

        return result

    except Exception as e:
        logger.error(f"Ошибка в MHK: {e}")
        raise


def generate_noise_vector(size, scale=1.0):
    """Генерация вектора шума с логированием"""
    logger.debug(f"Генерация шума. Размер: {size}, масштаб: {scale}")
    try:
        noise = np.random.rand(size) - 0.5
        noise = noise * scale
        logger.debug(f"Шум сгенерирован. Диапазон: [{noise.min()}, {noise.max()}]")
        return noise
    except Exception as e:
        logger.error(f"Ошибка генерации шума: {e}")
        raise


def add_noise(signal, noise):
    """Добавление шума к сигналу с логированием"""
    logger.debug("Добавление шума к сигналу")
    try:
        logger.debug(f"Сигнал shape: {signal.shape}, шум shape: {noise.shape}")
        noisy_signal = signal + noise.reshape(signal.shape)
        logger.debug("Шум успешно добавлен")
        return noisy_signal
    except Exception as e:
        logger.error(f"Ошибка добавления шума: {e}")
        raise


def signal_to_noise_ratio(signal, noise):
    """Вычисление SNR с логированием"""
    logger.debug("Вычисление SNR")
    try:
        signal_power = np.sum(signal ** 2)
        noise_power = np.sum(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        logger.info(f"SNR вычислен: {snr:.2f} дБ")
        return snr
    except Exception as e:
        logger.error(f"Ошибка вычисления SNR: {e}")
        raise


def error_occurred(x_ref, x_noisy, threshold=0.1):
    """Проверка ошибки с логированием"""
    logger.debug("Проверка наличия ошибки")
    try:
        error_norm = np.linalg.norm(x_ref - x_noisy)
        logger.debug(f"Норма ошибки: {error_norm}, порог: {threshold}")
        result = error_norm > threshold
        logger.info(f"Ошибка {'обнаружена' if result else 'не обнаружена'}")
        return result
    except Exception as e:
        logger.error(f"Ошибка проверки ошибки: {e}")
        raise